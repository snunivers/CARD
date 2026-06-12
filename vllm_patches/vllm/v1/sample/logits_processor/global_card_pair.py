# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import paired_logits
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

_GLOBAL_CARD_PAIR_ID_KEYS = (
    "global_card_pair_id",
    "logits_pair_id",
)
_GLOBAL_CARD_ROLE_KEYS = (
    "global_card_role",
    "logits_pair_role",
)

_GLOBAL_CARD_SUPPORT_FULL_VOCAB = "full_vocab"
_GLOBAL_CARD_SUPPORT_MAIN_AUX_TOPK_UNION = "main_aux_topk_union"
_GLOBAL_CARD_SUPPORT_MODES = {
    _GLOBAL_CARD_SUPPORT_FULL_VOCAB,
    _GLOBAL_CARD_SUPPORT_MAIN_AUX_TOPK_UNION,
}


@dataclass(frozen=True)
class _GlobalCARDPairInfo:
    pair_id: str
    role: str
    dynamic_strength_max: float
    main_bias_coeff: float
    direction_sign: int
    support_mode: str
    support_top_k: int
    trace_path: str | None
    output_ids: list[int]


class GlobalCARDPairLogitsProcessor(LogitsProcessor):
    """Batch-level Global CARD logits fusion for paired vLLM requests.

    The processor is intentionally narrow and greedy-only. It expects each
    logical sample to be submitted as paired main/base requests. The scheduler
    keeps requests with logits_pair_id/logits_pair_role together at
    sample-producing steps. Here "base" is the empty-document auxiliary branch.
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args or {}
        pair_info = paired_logits.get_paired_logits_pair_info(
            extra_args,
            pair_id_keys=_GLOBAL_CARD_PAIR_ID_KEYS,
            role_keys=_GLOBAL_CARD_ROLE_KEYS,
        )
        if pair_info is None:
            return None

        if sampling_params.temperature != 0.0:
            raise ValueError("Global CARD vLLM only supports greedy temperature=0.0")

        dynamic_strength_max = float(
            extra_args.get("global_card_dynamic_strength_max", 1.0)
        )
        if not math.isfinite(dynamic_strength_max) or dynamic_strength_max <= 0.0:
            raise ValueError(
                "Global CARD vLLM dynamic_strength_max must be a positive finite value"
            )
        main_bias_coeff = float(extra_args.get("global_card_main_bias_coeff", 0.0))
        if not math.isfinite(main_bias_coeff) or not (
            -1.0 <= main_bias_coeff <= 1.0
        ):
            raise ValueError(
                "Global CARD vLLM main_bias_coeff must be a finite value in [-1, 1]"
            )
        direction_sign = int(extra_args.get("global_card_direction_sign", 1))
        if direction_sign not in {1, -1}:
            raise ValueError("Global CARD vLLM direction_sign must be 1 or -1")
        support_mode = str(
            extra_args.get("global_card_support_mode", _GLOBAL_CARD_SUPPORT_FULL_VOCAB)
        )
        if support_mode not in _GLOBAL_CARD_SUPPORT_MODES:
            raise ValueError(
                "Global CARD vLLM support_mode must be one of "
                f"{sorted(_GLOBAL_CARD_SUPPORT_MODES)}"
            )
        support_top_k = int(extra_args.get("global_card_support_top_k", 10))
        if (
            support_mode == _GLOBAL_CARD_SUPPORT_MAIN_AUX_TOPK_UNION
            and support_top_k <= 0
        ):
            raise ValueError(
                "Global CARD vLLM support_top_k must be positive when "
                "support_mode=main_aux_topk_union"
            )
        return None

    def __init__(
        self, vllm_config: Any, device: torch.device, is_pin_memory: bool
    ) -> None:
        del vllm_config, is_pin_memory
        self.device = device
        self.req_info: dict[int, _GlobalCARDPairInfo] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int] | None,
        output_ids: list[int],
    ) -> _GlobalCARDPairInfo | None:
        del prompt_ids
        extra_args = params.extra_args or {}
        pair_info = paired_logits.get_paired_logits_pair_info(
            extra_args,
            pair_id_keys=_GLOBAL_CARD_PAIR_ID_KEYS,
            role_keys=_GLOBAL_CARD_ROLE_KEYS,
        )
        if pair_info is None:
            return None
        pair_id, role = pair_info
        dynamic_strength_max = float(
            extra_args.get("global_card_dynamic_strength_max", 1.0)
        )
        main_bias_coeff = float(extra_args.get("global_card_main_bias_coeff", 0.0))
        direction_sign = int(extra_args.get("global_card_direction_sign", 1))
        support_mode = str(
            extra_args.get("global_card_support_mode", _GLOBAL_CARD_SUPPORT_FULL_VOCAB)
        )
        support_top_k = int(extra_args.get("global_card_support_top_k", 10))
        trace_path_value = extra_args.get("global_card_trace_path")
        trace_path = str(trace_path_value) if trace_path_value else None
        return _GlobalCARDPairInfo(
            pair_id=str(pair_id),
            role=str(role),
            dynamic_strength_max=dynamic_strength_max,
            main_bias_coeff=main_bias_coeff,
            direction_sign=direction_sign,
            support_mode=support_mode,
            support_top_k=support_top_k,
            trace_path=trace_path,
            output_ids=output_ids,
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_info, batch_update, self._new_state)

    @staticmethod
    def _compute_dynamic_strength(
        main_scores: torch.Tensor,
        aux_scores: torch.Tensor,
        dynamic_strength_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_probs = F.softmax(main_scores.float(), dim=-1)
        aux_probs = F.softmax(aux_scores.float(), dim=-1)
        safe_main_probs = torch.clamp(main_probs, min=1e-12)
        safe_aux_probs = torch.clamp(aux_probs, min=1e-12)
        kappa = torch.sum(
            safe_aux_probs * (torch.log(safe_aux_probs) - torch.log(safe_main_probs))
        )
        gamma = 1.0 / float(dynamic_strength_max)
        strength = 1.0 / torch.log(torch.exp(kappa.new_tensor(gamma)) + kappa)
        return strength, kappa

    @staticmethod
    def _write_trace_record(
        trace_path: str | None,
        record: dict[str, int | float],
    ) -> None:
        if not trace_path:
            return
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    @classmethod
    def _apply_pair(
        cls,
        main_scores: torch.Tensor,
        aux_scores: torch.Tensor,
        info: _GlobalCARDPairInfo,
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        safe_main_scores = torch.nan_to_num(main_scores.float(), nan=0.0)
        safe_aux_scores = torch.nan_to_num(aux_scores.float(), nan=0.0)
        dynamic_strength, kappa = cls._compute_dynamic_strength(
            main_scores=safe_main_scores,
            aux_scores=safe_aux_scores,
            dynamic_strength_max=info.dynamic_strength_max,
        )
        sign = float(info.direction_sign)
        fused = (
            (1.0 - abs(info.main_bias_coeff)) * safe_main_scores
            + sign * dynamic_strength * (safe_main_scores - safe_aux_scores)
        )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        if info.support_mode == _GLOBAL_CARD_SUPPORT_MAIN_AUX_TOPK_UNION:
            top_k = min(int(info.support_top_k), fused.shape[-1])
            _, main_top_ids = torch.topk(safe_main_scores, k=top_k)
            _, aux_top_ids = torch.topk(safe_aux_scores, k=top_k)
            support_mask = torch.zeros_like(fused, dtype=torch.bool)
            support_mask[main_top_ids] = True
            support_mask[aux_top_ids] = True
            fused = fused.masked_fill(~support_mask, float("-inf"))
        trace_record = {
            "position": int(len(info.output_ids)),
            "kl_aux_vs_main": float(kappa.detach().float().cpu().item()),
            "strength": float(dynamic_strength.detach().float().cpu().item()),
        }
        return fused.to(main_scores.dtype), trace_record

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        pairs: dict[str, dict[str, tuple[int, _GlobalCARDPairInfo]]] = {}
        active_sample_indices = (
            paired_logits.get_paired_logits_active_sample_indices()
        )
        for req_idx, info in self.req_info.items():
            if req_idx >= logits.shape[0]:
                continue
            if (
                active_sample_indices is not None
                and req_idx not in active_sample_indices
            ):
                continue
            pairs.setdefault(info.pair_id, {})[info.role] = (req_idx, info)

        if not pairs:
            return logits

        for pair_id, roles in pairs.items():
            if "main" not in roles or "base" not in roles:
                raise RuntimeError(
                    "vLLM paired logits scheduling constraint failed: "
                    f"pair_id={pair_id!r} is missing main/base in this sampler batch"
                )

            main_idx, main_info = roles["main"]
            base_idx, _ = roles["base"]
            fused_scores, trace_record = self._apply_pair(
                logits[main_idx],
                logits[base_idx],
                main_info,
            )
            next_token_id = int(torch.argmax(fused_scores).item())
            trace_record["output_token_id"] = next_token_id
            self._write_trace_record(main_info.trace_path, trace_record)

            logits[main_idx].fill_(float("-inf"))
            logits[base_idx].fill_(float("-inf"))
            logits[main_idx, next_token_id] = 0.0
            logits[base_idx, next_token_id] = 0.0

        return logits

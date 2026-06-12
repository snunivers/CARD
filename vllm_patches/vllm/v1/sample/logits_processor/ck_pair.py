# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import paired_logits
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor


def set_ck_active_sample_indices(indices: list[int] | None) -> None:
    paired_logits.set_paired_logits_active_sample_indices(indices)


@dataclass(frozen=True)
class _CKPairInfo:
    pair_id: str
    role: str
    alpha: float
    adaptive: bool
    select_top: int
    relative_top: float


class CKPairLogitsProcessor(LogitsProcessor):
    """Batch-level CK-PLUG logits fusion for paired vLLM requests.

    This processor is inert for ordinary requests. A request participates when
    SamplingParams.extra_args contains logits_pair_id/logits_pair_role. Legacy
    ck_pair_id/ck_role keys are still accepted for old runs.
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args or {}
        pair_info = paired_logits.get_paired_logits_pair_info(extra_args)
        if pair_info is None:
            return None

        _, role = pair_info
        if role not in {"main", "base"}:
            raise ValueError("vLLM paired logits role must be 'main' or 'base'")

        if sampling_params.temperature != 0.0:
            raise ValueError("CK vLLM only supports greedy temperature=0.0")

        select_top = int(extra_args.get("ck_select_top", 10))
        relative_top = float(extra_args.get("ck_relative_top", 0.01))
        alpha = float(extra_args.get("ck_alpha", 0.0))
        if select_top <= 0:
            raise ValueError("CK vLLM ck_select_top must be positive")
        if not (0.0 < relative_top <= 1.0):
            raise ValueError("CK vLLM ck_relative_top must be in (0, 1]")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("CK vLLM ck_alpha must be in [0, 1]")
        return None

    def __init__(
        self, vllm_config: Any, device: torch.device, is_pin_memory: bool
    ) -> None:
        del vllm_config, is_pin_memory
        self.device = device
        self.req_info: dict[int, _CKPairInfo] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int] | None,
        output_ids: list[int],
    ) -> _CKPairInfo | None:
        del prompt_ids, output_ids
        extra_args = params.extra_args or {}
        pair_info = paired_logits.get_paired_logits_pair_info(extra_args)
        if pair_info is None:
            return None
        pair_id, role = pair_info
        return _CKPairInfo(
            pair_id=str(pair_id),
            role=str(role),
            alpha=float(extra_args.get("ck_alpha", 0.0)),
            adaptive=bool(extra_args.get("ck_adaptive", False)),
            select_top=int(extra_args.get("ck_select_top", 10)),
            relative_top=float(extra_args.get("ck_relative_top", 0.01)),
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_info, batch_update, self._new_state)

    @staticmethod
    def _relative_top_filter(
        main_scores: torch.Tensor,
        base_scores: torch.Tensor,
        select_top: int,
        relative_top: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_log_probs = F.log_softmax(main_scores, dim=-1)
        base_log_probs = F.log_softmax(base_scores, dim=-1)

        vocab_size = main_log_probs.shape[-1]
        keep_count = min(max(int(select_top), 1), vocab_size)
        main_top_values = torch.topk(main_log_probs, keep_count, dim=-1).values
        base_top_values = torch.topk(base_log_probs, keep_count, dim=-1).values

        rel_log = math.log(float(relative_top))
        main_thresh = torch.minimum(
            main_top_values[:, -1],
            torch.max(main_log_probs, dim=-1).values + rel_log,
        ).unsqueeze(-1)
        base_thresh = torch.minimum(
            base_top_values[:, -1],
            torch.max(base_log_probs, dim=-1).values + rel_log,
        ).unsqueeze(-1)

        mask = (main_log_probs < main_thresh) & (base_log_probs < base_thresh)
        filtered_main = main_scores.clone()
        filtered_base = base_scores.clone()
        filtered_main[mask] = -1e10
        filtered_base[mask] = -1e10
        return filtered_main, filtered_base, mask

    @classmethod
    def _apply_pair(
        cls,
        main_scores: torch.Tensor,
        base_scores: torch.Tensor,
        info: _CKPairInfo,
    ) -> torch.Tensor:
        filtered_main, filtered_base, mask = cls._relative_top_filter(
            main_scores=main_scores.float().unsqueeze(0),
            base_scores=base_scores.float().unsqueeze(0),
            select_top=info.select_top,
            relative_top=info.relative_top,
        )
        filtered_main = filtered_main.squeeze(0)
        filtered_base = filtered_base.squeeze(0)
        mask = mask.squeeze(0)

        probs_main = F.softmax(filtered_main, dim=-1)
        probs_base = F.softmax(filtered_base, dim=-1)
        entropy_main = -torch.sum(probs_main * torch.log(probs_main + 1e-9))
        entropy_base = -torch.sum(probs_base * torch.log(probs_base + 1e-9))

        info_gain = entropy_base - entropy_main
        is_adjust = (info_gain - torch.abs(entropy_main)) < 0
        if not bool(is_adjust.item()):
            return filtered_main.to(main_scores.dtype)

        base_for_context = filtered_base.clone()
        base_for_context[mask] = -1e3
        logits_context = filtered_main - base_for_context
        filtered_base[mask] = -1e10

        if info.adaptive:
            diff = torch.abs(entropy_main - entropy_base)
            entropy_sum = torch.clamp(entropy_main + entropy_base, min=1e-6)
            normalization_factor = 1.0 + (diff / entropy_sum)
            denominator = torch.clamp(
                entropy_main + entropy_base * normalization_factor,
                min=1e-6,
            )
            fused = (
                2.0 * filtered_base * entropy_main / denominator
                + 2.0 * logits_context * entropy_base / denominator
            )
        else:
            fused = info.alpha * filtered_base + (1.0 - info.alpha) * logits_context

        return fused.to(main_scores.dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        pairs: dict[str, dict[str, tuple[int, _CKPairInfo]]] = {}
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
            fused_scores = self._apply_pair(
                logits[main_idx],
                logits[base_idx],
                main_info,
            )
            next_token_id = int(torch.argmax(fused_scores).item())

            logits[main_idx].fill_(float("-inf"))
            logits[base_idx].fill_(float("-inf"))
            logits[main_idx, next_token_id] = 0.0
            logits[base_idx, next_token_id] = 0.0

        return logits

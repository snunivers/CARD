# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

PAIR_ID_KEYS = ("logits_pair_id", "ck_pair_id")
PAIR_ROLE_KEYS = ("logits_pair_role", "ck_role")
PAIR_ROLES = {"main", "base"}

_active_sample_indices: set[int] | None = None


def set_paired_logits_active_sample_indices(indices: list[int] | None) -> None:
    global _active_sample_indices
    _active_sample_indices = None if indices is None else set(indices)


def get_paired_logits_active_sample_indices() -> set[int] | None:
    return _active_sample_indices


def _first_present(extra_args: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in extra_args:
            return extra_args[key]
    return None


def get_paired_logits_pair_info(
    extra_args: dict[str, Any] | None,
    *,
    pair_id_keys: tuple[str, ...] = PAIR_ID_KEYS,
    role_keys: tuple[str, ...] = PAIR_ROLE_KEYS,
) -> tuple[str, str] | None:
    extra_args = extra_args or {}
    pair_id = _first_present(extra_args, pair_id_keys)
    role = _first_present(extra_args, role_keys)
    if pair_id is None and role is None:
        return None
    if pair_id is None or role not in PAIR_ROLES:
        raise ValueError(
            "vLLM paired logits requests must provide a pair id and "
            "role in {'main', 'base'}"
        )
    return str(pair_id), str(role)

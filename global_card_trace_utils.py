import json
import typing as t


def sanitize_global_card_token_text(token_text: str) -> str:
    return token_text.replace("\n", "\\n")


def read_jsonl_records(path: str) -> t.List[t.Dict[str, t.Any]]:
    records: t.List[t.Dict[str, t.Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _first_present(
    mapping: t.Mapping[str, t.Any],
    keys: t.Iterable[str],
) -> t.Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_float(value: t.Any) -> t.Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: t.Any) -> t.Optional[int]:
    if value is None:
        return None
    return int(value)


def normalize_global_card_token_trace(
    trace: t.Optional[t.Sequence[t.Mapping[str, t.Any]]],
) -> t.List[t.Dict[str, t.Any]]:
    if not trace:
        return []

    normalized_trace: t.List[t.Dict[str, t.Any]] = []
    for index, record in enumerate(trace):
        alpha_value = _first_present(
            record,
            (
                "alpha_t",
                "dynamic_alpha_t",
                "dynamic_alpha",
                "strength",
            ),
        )
        normalized_trace.append({
            "position": index,
            "output_token_id": int(record["output_token_id"]),
            "output_token_text": record["output_token_text"],
            "kl_aux_vs_main": _optional_float(record.get("kl_aux_vs_main")),
            "alpha_t": _optional_float(alpha_value),
        })
    return normalized_trace


def build_global_card_token_trace(
    tokenizer,
    output_token_ids: t.Sequence[int],
    source_records: t.Sequence[t.Mapping[str, t.Any]],
) -> t.List[t.Dict[str, t.Any]]:
    output_token_count = len(output_token_ids)
    source_record_count = len(source_records)
    if source_record_count < output_token_count:
        raise ValueError(
            "Global CARD token trace has fewer records than output tokens: "
            f"trace_records={source_record_count}, output_tokens={output_token_count}"
        )

    trace: t.List[t.Dict[str, t.Any]] = []
    source_index = 0

    for index in range(output_token_count):
        output_token_id = output_token_ids[index]
        try:
            output_token_id_int = int(output_token_id)
        except (TypeError, ValueError):
            output_token_id_int = int(output_token_ids[index])

        source_record = None
        while source_index < source_record_count:
            candidate_record = source_records[source_index]
            source_index += 1
            source_output_token_id = _optional_int(
                _first_present(candidate_record, ("output_token_id",))
            )
            if (
                source_output_token_id is None
                or source_output_token_id == output_token_id_int
            ):
                source_record = candidate_record
                break

        if source_record is None:
            raise ValueError(
                "Global CARD token trace could not find a record matching the output sequence: "
                f"output_token_id={output_token_id_int}, index={index}, "
                f"trace_records={source_record_count}, output_tokens={output_token_count}"
            )

        kl_value = _first_present(
            source_record,
            (
                "kl_aux_vs_main",
                "current_step_kappa_aux_vs_main",
                "applied_kappa_aux_vs_main",
            ),
        )
        alpha_value = _first_present(
            source_record,
            (
                "alpha_t",
                "dynamic_alpha_t",
                "dynamic_alpha",
                "strength",
            ),
        )

        trace.append({
            "position": index,
            "output_token_id": output_token_id_int,
            "output_token_text": sanitize_global_card_token_text(
                tokenizer.decode([output_token_id_int])
            ),
            "kl_aux_vs_main": (
                float(kl_value) if kl_value is not None else None
            ),
            "alpha_t": (
                float(alpha_value) if alpha_value is not None else None
            ),
        })

    return trace

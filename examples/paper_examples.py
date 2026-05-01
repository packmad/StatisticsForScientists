"""
Worked examples for every scenario used in the paper.

Run with:
    uv run examples

Optional flags:
    uv run examples --show-data
    uv run examples --json
"""

from __future__ import annotations

import json
import argparse
from typing import Any

import stats4science as stats

DATASETS: dict[str, dict[str, list[float]]] = {
    "reaction_time_ms": {
        "interface_a": [398, 410, 405, 392, 430, 415, 401, 389, 418, 407, 395, 423],
        "interface_b": [412, 439, 421, 445, 433, 427, 416, 438, 450, 429, 420],
    },
    "usability_ratings": {
        "old_version": [3, 4, 4, 3, 5, 4, 3, 4, 2, 3, 4, 3],
        "new_version": [4, 5, 4, 4, 5, 4, 5, 4, 4, 5, 3, 4],
    },
    "study_time_exam_scores": {
        "study_hours": [2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12],
        "exam_scores": [58, 66, 59, 69, 70, 66, 78, 72, 82, 75, 86, 83, 90, 88, 92],
    },
    "stress_sleep": {
        "stress_level": [2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
        "sleep_quality": [8.4, 8.0, 8.2, 7.5, 7.4, 7.6, 6.8, 7.0, 6.6, 6.3, 6.8, 5.9, 5.8, 5.5, 5.6],
    },
    "blood_pressure_mmhg": {
        "drug_group": [118, 121, 116, 123, 119, 117, 122, 120, 115, 124, 118, 121, 117, 119],
        "placebo_group": [124, 130, 127, 132, 125, 129, 126, 131, 128, 133, 125, 130, 127],
    },
    "teaching_confidence_ratings": {
        "method_a": [5, 6, 6, 5, 7, 6, 5, 6, 7, 5, 6, 6, 5, 7],
        "method_b": [4, 5, 5, 6, 6, 4, 5, 5, 5, 4, 5, 5, 5, 4],
    },
    "temperature_heart_rate": {
        "body_temperature_c": [37.0, 37.2, 37.1, 37.4, 37.3, 37.5, 37.6, 37.7, 37.8, 38.0, 38.1, 38.2],
        "heart_rate_bpm": [88, 93, 89, 94, 92, 97, 95, 104, 99, 106, 108, 109],
    },
}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def build_examples() -> list[dict[str, Any]]:
    reaction = DATASETS["reaction_time_ms"]
    usability = DATASETS["usability_ratings"]
    study = DATASETS["study_time_exam_scores"]
    stress_sleep = DATASETS["stress_sleep"]
    blood_pressure = DATASETS["blood_pressure_mmhg"]
    teaching = DATASETS["teaching_confidence_ratings"]
    temperature = DATASETS["temperature_heart_rate"]

    reaction_mean = stats.compare_independent_groups(
        reaction["interface_a"],
        reaction["interface_b"],
        estimand="mean_difference",
    )
    reaction_faster = stats.compare_independent_groups(
        reaction["interface_b"],
        reaction["interface_a"],
        estimand="stochastic_dominance",
    )
    usability_result = stats.compare_independent_groups(
        usability["new_version"],
        usability["old_version"],
        estimand="stochastic_dominance",
    )
    study_pearson = stats.correlation(
        study["study_hours"],
        study["exam_scores"],
        method="pearson",
    )
    study_spearman = stats.correlation(
        study["study_hours"],
        study["exam_scores"],
        method="spearman",
    )
    stress_spearman = stats.correlation(
        stress_sleep["stress_level"],
        stress_sleep["sleep_quality"],
        method="spearman",
    )
    blood_pressure_result = stats.compare_independent_groups(
        blood_pressure["drug_group"],
        blood_pressure["placebo_group"],
        estimand="mean_difference",
    )
    teaching_result = stats.compare_independent_groups(
        teaching["method_a"],
        teaching["method_b"],
        estimand="stochastic_dominance",
    )
    temperature_result = stats.correlation(
        temperature["body_temperature_c"],
        temperature["heart_rate_bpm"],
        method="pearson",
    )

    return [
        {
            "id": "EX-RT-01",
            "slug": "reaction_time_mean_difference",
            "title": "Reaction-time experiment",
            "question": "How many milliseconds faster is interface A than interface B on average?",
            "dataset": "reaction_time_ms",
            "result": reaction_mean,
            "report": stats.report_two_group(reaction_mean, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-OR-01",
            "slug": "usability_ratings",
            "title": "Ordinal usability rating study",
            "question": "Does the new app version tend to receive higher usability ratings?",
            "dataset": "usability_ratings",
            "result": usability_result,
            "report": stats.report_two_group(usability_result, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-CO-01",
            "slug": "study_time_exam_pearson",
            "title": "Study time and exam performance",
            "question": "Are study hours and exam scores linearly associated?",
            "dataset": "study_time_exam_scores",
            "result": study_pearson,
            "report": stats.report_correlation(study_pearson, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-CO-02",
            "slug": "study_time_exam_spearman",
            "title": "Study time and exam performance (rank-based)",
            "question": "Do study hours and exam scores show a monotonic association?",
            "dataset": "study_time_exam_scores",
            "result": study_spearman,
            "report": stats.report_correlation(study_spearman, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-BS-01",
            "slug": "stress_sleep_spearman",
            "title": "Stress level and sleep quality",
            "question": "What is the monotonic association between stress and sleep quality?",
            "dataset": "stress_sleep",
            "result": stress_spearman,
            "report": stats.report_correlation(stress_spearman, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-TG-02",
            "slug": "blood_pressure_welch",
            "title": "Blood-pressure treatment example",
            "question": "How much lower is post-treatment blood pressure in the drug group?",
            "dataset": "blood_pressure_mmhg",
            "result": blood_pressure_result,
            "report": stats.report_two_group(blood_pressure_result, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-OR-02",
            "slug": "teaching_methods",
            "title": "Teaching-method confidence ratings",
            "question": "Does teaching method A tend to produce higher confidence ratings?",
            "dataset": "teaching_confidence_ratings",
            "result": teaching_result,
            "report": stats.report_two_group(teaching_result, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-CO-03",
            "slug": "temperature_heart_rate",
            "title": "Body temperature and heart rate",
            "question": "Are body temperature and heart rate positively associated?",
            "dataset": "temperature_heart_rate",
            "result": temperature_result,
            "report": stats.report_correlation(temperature_result, digits=2, include_interpretation=False),
        },
        {
            "id": "EX-TG-01",
            "slug": "same_data_different_estimand",
            "title": "Same interface data, different estimand",
            "question": "Using the same reaction-time data, how often is interface A faster than interface B?",
            "dataset": "reaction_time_ms",
            "result": reaction_faster,
            "report": (
                stats.report_two_group(reaction_faster, digits=2, include_interpretation=False)
                + "\nInterpreted on the completion-time scale, this estimates that interface A is faster than "
                + f"interface B about {100.0 * reaction_faster.estimate:.1f}% of the time."
            ),
        },
    ]


def _examples_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for example in build_examples():
        result = example["result"]
        payload.append(
            {
                "id": example["id"],
                "slug": example["slug"],
                "title": example["title"],
                "question": example["question"],
                "dataset": example["dataset"],
                "report": example["report"],
                "result": _to_builtin(result.to_dict()),
            }
        )
    return payload


def print_examples(*, show_data: bool) -> None:
    for example in build_examples():
        print(f"### {example['id']} - {example['title']} (slug: {example['slug']})")
        print(f"Question: {example['question']}")
        print(f"Dataset: {example['dataset']}")
        if show_data:
            for key, values in DATASETS[example["dataset"]].items():
                print(f"  {key} = {values}")
        print(example["report"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all worked examples used in the paper.")
    parser.add_argument("--show-data", action="store_true", help="Print the raw data arrays for each example.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results for all paper examples.")
    args = parser.parse_args()

    if args.json:
        print(json.dumps({"datasets": DATASETS, "examples": _examples_payload()}, indent=2))
        return

    print_examples(show_data=args.show_data)


if __name__ == "__main__":
    main()

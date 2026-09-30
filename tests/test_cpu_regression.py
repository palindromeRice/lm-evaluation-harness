"""Small, deterministic CPU evaluations and results-table snapshots for #1883."""

import copy
import json
from pathlib import Path

import pytest
import torch

from lm_eval import evaluator
from lm_eval.api.group import Group
from lm_eval.config.group import AggMetricConfig
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table


REFERENCE_DIR = Path(__file__).parent / "testdata" / "cpu_regression"
EVAL_KWARGS = {
    "model": "hf",
    "model_args": {
        "pretrained": "EleutherAI/pythia-14m-deduped",
        "revision": "7386d9a4ae45aef494a6e704910394def3037fc5",
        "dtype": "float32",
    },
    "tasks": [
        "arc_easy",
        "arc_challenge",
        "lambada_openai",
        "wikitext",
        "mmlu_abstract_algebra",
    ],
    "device": "cpu",
    "batch_size": 1,
    "num_fewshot": 0,
    "limit": 10,
    "bootstrap_iters": 0,
    "log_samples": False,
    "random_seed": 0,
    "numpy_random_seed": 0,
    "torch_random_seed": 0,
    "fewshot_random_seed": 0,
}


def evaluate_cpu():
    """Run the reference configuration, including a small MMLU task group."""
    # Avoid CPU oversubscription under the existing pytest-xdist CI runner.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        kwargs = copy.deepcopy(EVAL_KWARGS)
        tasks = TaskManager().load(kwargs.pop("tasks"))["tasks"]
        group = Group(
            name="mmlu_regression",
            alias="MMLU (subset)",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        group.add(tasks.pop("mmlu_abstract_algebra"))
        results = evaluator.simple_evaluate(tasks=[*tasks.values(), group], **kwargs)
        assert results is not None
        return results
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def cpu_results():
    return evaluate_cpu()


@pytest.fixture(scope="module")
def cpu_reference():
    return json.loads((REFERENCE_DIR / "scores.json").read_text(encoding="utf-8"))


def test_cpu_scores(cpu_results, cpu_reference):
    assert cpu_reference["evaluation"] == EVAL_KWARGS
    assert set(cpu_results["n-samples"]) == {
        "arc_easy",
        "arc_challenge",
        "lambada_openai",
        "wikitext",
        "mmlu_abstract_algebra",
    }
    for counts in cpu_results["n-samples"].values():
        assert counts["effective"] == 10
    for column in ("results", "groups"):
        actual = cpu_results[column]
        expected = cpu_reference[column]
        assert actual.keys() == expected.keys()
        for task, metrics in expected.items():
            assert actual[task].keys() == metrics.keys(), task
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Accuracy changes by at least 0.1 at limit=10; continuous
                    # metrics allow only small floating-point implementation noise.
                    tolerance = 0 if metric.startswith(("acc,", "acc_norm,")) else 1e-4
                    assert actual[task][metric] == pytest.approx(
                        value, rel=tolerance, abs=0
                    ), (task, metric)
                else:
                    assert actual[task][metric] == value, (task, metric)


@pytest.mark.parametrize("column", ["results", "groups"])
def test_cpu_tables(cpu_results, cpu_reference, column):
    # Scores have their own assertions. Use reference numbers here so permitted
    # floating-point noise cannot cause changes in rounding or column widths.
    results = copy.deepcopy(cpu_results)
    for task, metrics in cpu_reference[column].items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                results[column][task][metric] = value
    expected = (REFERENCE_DIR / f"{column}.txt").read_text(encoding="utf-8")
    assert make_table(results, column=column) == expected

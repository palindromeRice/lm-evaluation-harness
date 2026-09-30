# CPU regression references

`tests/test_cpu_regression.py` evaluates the first ten examples of ARC Easy,
ARC Challenge, LAMBADA OpenAI, WikiText, and MMLU abstract algebra. It uses the
same small Pythia model as the existing evaluator tests, with a pinned model
revision, float32 on CPU, batch size one, zero-shot prompts, fixed seeds, and no
bootstrap statistics. A one-subject MMLU group also exercises aggregation and
group table rendering.

Run from the repository root after installing the existing `dev` and `hf` extras:

```sh
python -m pytest tests/test_cpu_regression.py -v
```

The existing CPU CI job discovers these tests automatically. The first run
downloads public model weights and datasets; subsequent runs use the Hugging Face
cache. No GPU or credentials are required. This is a small regression check, not
a measurement of full benchmark performance or coverage of every model backend.

`scores.json` contains raw task and group scores, the evaluation configuration,
and dependency versions used to generate the reference. Accuracy must match
exactly: with ten examples, one changed answer changes it by 0.1. Continuous
metrics allow relative error of `1e-4` for floating-point implementation noise.
The table tests substitute reference numbers into the current evaluation result
before rendering, then compare the complete output, including whitespace, column
widths, ordering, aliases, task versions, filters, and group indentation. This
keeps formatting checks independent of numerical rounding differences.

## Updating references

Investigate a failure before updating references. Check changes to task prompts,
data, model behavior, dependencies, and table rendering. Review all score and
table differences; do not regenerate references merely to make a failure pass.
Dataset revisions are not pinned, matching existing task tests, so upstream data
changes may require investigation too. Dependency versions are provenance, not
a reason to skip regression checks.

For an intentional change, run this from the repository root in the same
environment used to verify the tests:

```python
import json
import runpy
from importlib.metadata import version

from lm_eval.utils import make_table

suite = runpy.run_path("tests/test_cpu_regression.py")
results = suite["evaluate_cpu"]()
directory = suite["REFERENCE_DIR"]
reference = {
    "evaluation": suite["EVAL_KWARGS"],
    "dependencies": {
        name: version(name)
        for name in ("torch", "transformers", "datasets", "pytablewriter")
    },
    "results": results["results"],
    "groups": results["groups"],
}
(directory / "scores.json").write_text(
    json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
for column in ("results", "groups"):
    (directory / f"{column}.txt").write_text(
        make_table(results, column=column), encoding="utf-8"
    )
```

Rerun the tests independently to confirm reproducibility. GPU coverage is deferred
because the repository currently has no active GPU CI job.

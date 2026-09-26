import pandas as pd
import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.scorer import Score

from misuse_bench.aggregate import (
    aggregate,
    ceilings,
    challenge_scores,
    pass_at_k,
    variant_scores,
)
from misuse_bench.config import ModelVariant


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [(4, 1, 2, 0.5), (4, 2, 2, 5 / 6), (4, 0, 4, 0), (4, 4, 2, 1), (4, 1, 4, 1)],
)
def test_pass_at_k(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "c", "k"), [(0, 0, 1), (2, 3, 1), (2, -1, 1), (2, 1, 3), (2, 1, 0)]
)
def test_invalid_pass_at_k(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_variant_scores_weight_challenges_equally():
    result = variant_scores({"first": [1, 0], "second": [0, 0, 0, 0]}, k=2)
    assert result == {
        "challenges": 2,
        "attempts": 6,
        "success_rate": 0.25,
        "k": 2,
        "pass_at_k": 0.5,
    }


def test_ceiling_chooses_whole_variants_per_metric():
    scores = pd.DataFrame(
        [
            {
                "base_model": "base",
                "model_id": "original",
                "success_rate": 0.5,
                "pass_at_k": 0.7,
                "k": 2,
            },
            {
                "base_model": "base",
                "model_id": "abliterated",
                "success_rate": 0.4,
                "pass_at_k": 0.8,
                "k": 2,
            },
            {
                "base_model": "other",
                "model_id": "fine-tuned",
                "success_rate": 0.2,
                "pass_at_k": 0.3,
                "k": 2,
            },
        ]
    )
    result = ceilings(scores).set_index("base_model")
    assert result.loc["base", "success_rate"] == 0.5
    assert result.loc["base", "success_rate_model_id"] == "original"
    assert result.loc["base", "pass_at_k"] == 0.8
    assert result.loc["base", "pass_at_k_model_id"] == "abliterated"
    assert result.loc["other", "variants"] == 1


def make_log(values):
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-09-26T00:00:00Z",
            task="cybench",
            model="mockllm/model",
            dataset=EvalDataset(),
            config=EvalConfig(epochs=len(values)),
        ),
        samples=[
            EvalSample(
                id="challenge",
                epoch=epoch,
                input="test",
                target="flag",
                scores={"includes": Score(value=value)},
            )
            for epoch, value in enumerate(values, start=1)
        ],
    )


def test_read_epoch_scores():
    assert challenge_scores(make_log(["C", "I", 1, 0])) == {"challenge": [1, 0, 1, 0]}


@pytest.mark.parametrize("value", ["P", 0.5, {"score": 1}])
def test_reject_nonbinary_scores(value):
    with pytest.raises(ValueError, match="Non-binary"):
        challenge_scores(make_log([value]))


def test_reject_missing_and_duplicate_scores():
    log = make_log(["C", "I"])
    log.samples[1].epoch = 1
    with pytest.raises(ValueError, match="Duplicate"):
        challenge_scores(log)
    log.samples[1].epoch = 2
    log.samples[1].scores = None
    with pytest.raises(ValueError, match="Missing valid score"):
        challenge_scores(log)


def test_reject_incomplete_epochs():
    log = make_log(["C", "I"])
    log.samples.pop()
    with pytest.raises(ValueError, match="Incomplete epochs"):
        challenge_scores(log)


def test_aggregate_latest_success_and_config_guard(tmp_path):
    model = ModelVariant(
        id="original",
        hf_id="owner/model",
        model="mockllm/model",
        base_model="base",
        variant="original",
    )
    directory = tmp_path / "logs" / model.id
    directory.mkdir(parents=True)
    for day, values, status in [
        (1, ["C", "C"], "success"),
        (2, ["C", "I"], "success"),
        (3, ["I", "I"], "error"),
    ]:
        log = make_log(values)
        log.eval.created = f"2026-09-{day:02}T00:00:00Z"
        log.eval.metadata = {"model_variant": model.model_dump(mode="json")}
        log.status = status
        # JSON logs are also supported by Inspect's public log API.
        write_eval_log(log, directory / f"2026-09-{day:02}T00-00-00_cybench_run.json")
    scores, ceiling = aggregate(
        {model.id: model},
        k=2,
        log_dir=tmp_path / "logs",
        results_dir=tmp_path / "results",
    )
    assert scores.success_rate.tolist() == [0.5]
    assert scores.pass_at_k.tolist() == [1.0]
    assert scores.log.iloc[0].endswith("2026-09-02T00-00-00_cybench_run.json")
    assert ceiling.success_rate_model_id.tolist() == ["original"]
    model.hf_id = "owner/changed"
    with pytest.raises(ValueError, match="config differs"):
        aggregate(
            {model.id: model},
            log_dir=tmp_path / "logs",
            results_dir=tmp_path / "results",
        )


def test_aggregate_rejects_different_challenge_sets(tmp_path):
    models = {}
    for variant in ("original", "abliterated"):
        model = ModelVariant(
            id=variant,
            hf_id=f"owner/{variant}",
            model="mockllm/model",
            base_model="base",
            variant=variant,
        )
        models[model.id] = model
        directory = tmp_path / "logs" / model.id
        directory.mkdir(parents=True)
        log = make_log(["C"])
        log.eval.metadata = {"model_variant": model.model_dump(mode="json")}
        log.samples[0].id = variant
        write_eval_log(log, directory / "2026-09-26T00-00-00_cybench_run.json")
    with pytest.raises(ValueError, match="Challenge sets differ"):
        aggregate(models, log_dir=tmp_path / "logs", results_dir=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_aggregate_rejects_no_logs_and_insufficient_epochs(tmp_path):
    with pytest.raises(ValueError, match="No successful"):
        aggregate({}, log_dir=tmp_path / "logs", results_dir=tmp_path / "results")
    with pytest.raises(ValueError, match="pass@k requires"):
        variant_scores({"challenge": [1]}, k=2)

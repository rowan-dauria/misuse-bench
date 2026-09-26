# misuse-bench

A minimal harness for evaluating original, abliterated, and fine-tuned open-weight
models on [Cybench](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/cybench).
It reports each base model's **ceiling**: its best score across the evaluated
variants. It uses the task supplied by `inspect_evals`, with the default `hard`
challenge variant; there is no abliteration code here.

Use Python 3.12. Real runs need a Linux machine with a vLLM-compatible GPU,
Docker running, and internet access to download model weights, challenge files,
and container images. Cybench gives the model a Kali Linux environment with
internet access and the ability to install software. Run it on a dedicated
evaluation machine. Cybench requires an explicit acknowledgment of these risks;
the harness never sets it for you.

```sh
uv sync --extra gpu
export CYBENCH_ACKNOWLEDGE_RISKS=1
uv run misuse-bench run --model qwen3-8b --limit 2 --epochs 3
uv run misuse-bench run --model qwen3-8b-abliterated --limit 2 --epochs 3
uv run misuse-bench aggregate --k 2
```

Remove `--limit` for the full benchmark. `run` also accepts `--message-limit`,
`--token-limit`, and `--max-connections`, forwarded to Inspect. Omitted limits
use Inspect/task defaults. Use the same challenge selection, epochs, and resource
budgets for variants you compare. The CLI defaults to `configs/models.yaml`,
`logs/`, and `results/`; override these with `--config`, `--log-dir`, and
`--results-dir` (aggregate only).

Add model entries to `configs/models.yaml` with a unique filesystem-safe `id`,
`hf_id`, `base_model`, `variant`, optional `release_date`, and `model_args`.
Variants of the same base model share `base_model`. Inspect forwards `model_args`
to `vllm serve` as CLI flags. The Qwen3 examples enable automatic tool choice and
use the `hermes` tool-call parser and `qwen3` reasoning parser. The abliterated
[checkpoint](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated) was verified on
2026-09-26; its date uses Hugging Face's repository creation date.

Logs are saved under `logs/<model_id>/`. Aggregation reads each configured
variant's latest successful log (by run creation time), skipping variants without
a successful run. It rejects missing, erroneous, duplicate, or non-binary epoch
scores, changed model configs, and different challenge sets within a base model.
Reruns are not pooled. The ceiling covers only variants with successful logs.

Outputs:

- `results/cybench_scores.csv`: one row per evaluated variant, with challenge and
  attempt counts, success rate, `k`, pass@k, and the source log path.
- `results/cybench_ceiling.csv`: one row per base model, with the best success rate
  and pass@k and the winning model id for each metric. Ties use the first id in
  alphabetical order. A winner is one whole variant, rather than a different
  variant chosen for each challenge.

Success rate is the mean of each challenge's fraction of successful epochs.
For a challenge with `n` epochs and `c` successes, pass@k is
`1 - C(n-c, k) / C(n, k)`, averaged equally across challenges. `k` defaults to 1
and must be no greater than the number of scored epochs for every challenge.
These epochs are separate attempts; retries inside Cybench's agent are part of
one attempt. Rates are fractions between 0 and 1.

For local checks without a GPU, Docker, or network access after installation:

```sh
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

The smoke test replaces Cybench with a tiny in-memory task and evaluates it with
`mockllm/model`, then reads the real Inspect logs and checks both CSVs.

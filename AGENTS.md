# Instructions for Agents

## Communication — STRICT

Give simple, concise answers to questions and when explaining code changes. Skip preamble and restating the question; state the answer and, if needed, the reason.
Use plain, everyday wording; explain only the detail needed to answer the user's question.

## Project

`misuse-bench` measures how capable jailbroken open-weight models are at
harmful tasks when someone misuses them. "Jailbroken" here mostly means
abliterated models: models whose refusal behavior has been removed by editing
their weights. The goal is to measure what these models can actually do once
they no longer refuse, not whether they refuse.

The package lives in `src/misuse_bench`. It runs Cybench from `inspect_evals`
through Inspect's API providers and aggregates epoch scores into variant scores
and each base model's best score across variants. Model variants are configured
in `configs/models.yaml`; logs go in `logs/` and CSVs in `results/`.

Use `uv sync` to install dependencies. Run `uv run ruff check`,
`uv run ruff format --check`, and `uv run pytest` before submitting changes.
Tests use an in-memory task and `mockllm/model`; they need no GPU or Docker.

## VM updates

For tracked repository changes needed on the evaluation VM, commit and push
locally, then pull on the VM. Do not copy source files, dependency files, or
tracked documentation to the VM with `gcloud compute scp`. Check both Git
worktrees before pulling, and preserve VM-only files such as `.env`, `GCP/`,
`.prompts/`, logs, and results. Never commit or transfer API keys.
Keep `.prompts/` private and out of Git. When a prompt set changes, give the
user commands to install it on the VM; do not copy it there directly.

## Python

- Use Python 3.12. Don't use features from later versions, and don't write
  workarounds for earlier ones.

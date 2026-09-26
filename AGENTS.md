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

The repo is at an early stage; there is no code yet beyond `README.md` and
`LICENSE`.

## Python

- Use Python 3.12. Don't use features from later versions, and don't write
  workarounds for earlier ones.

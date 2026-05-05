# Coverage Baseline (collected 2026-05-04)

This document records the unit-test line coverage of every package in this
monorepo at the time the coverage CI was first introduced. New thresholds will
be raised as we close the gap toward our long-term targets (see "Trajectory"
below).

How the numbers were collected:

- `uv sync --group test` in each package, then `pytest --cov=<package>` against
  `tests/unit_tests/` only (`tests/` for `text-splitters`, which has no
  separate unit/integration split for runnable tests).
- `pytest-cov` was installed ad-hoc where it wasn't yet a declared test
  dependency. This PR adds it explicitly.
- No Docker services, network, or vendor API keys were available, so any test
  that requires them was skipped — these baselines reflect the offline
  default `pytest` run.

## Core packages

| Package              | Coverage % | Lines Covered | Lines Missing | Notes                                                                  |
|----------------------|-----------:|--------------:|--------------:|------------------------------------------------------------------------|
| libs/core            |        83% |        16,447 |         3,439 | `pytest --cov=langchain_core tests/unit_tests/`. 19,886 statements.    |
| libs/langchain_v1    |        91% |         3,228 |           309 | `pytest --cov=langchain ...`. 786 passed, 13 skipped (Docker-only).    |
| libs/langchain       |        74% |        15,048 |         5,387 | `pytest --cov=langchain_classic ...`. Source package is `langchain_classic`. |
| libs/text-splitters  |        79% |           867 |           226 | `pytest --cov=langchain_text_splitters tests/unit_tests/`.             |

## Partner packages

Each partner is measured against its own `langchain_<name>` package only
(`pytest --cov=langchain_<name>`). Tests requiring real API keys or network
access are skipped and **not** included here, so partner coverage will look
artificially low for thin API-wrapper packages.

| Package                       | Coverage % | Lines Covered | Lines Missing | Notes                                            |
|-------------------------------|-----------:|--------------:|--------------:|--------------------------------------------------|
| libs/partners/anthropic       |        66% |         1,212 |           622 | 191 passed.                                      |
| libs/partners/chroma          |        43% |           133 |           173 | 29 passed.                                       |
| libs/partners/deepseek        |        86% |           118 |            19 | 26 passed, 1 skipped.                            |
| libs/partners/exa             |        80% |            86 |            22 | 2 passed.                                        |
| libs/partners/fireworks       |        73% |           430 |           157 | 55 passed.                                       |
| libs/partners/groq            |        62% |           330 |           202 | 56 passed, 1 skipped.                            |
| libs/partners/huggingface     |        50% |           509 |           509 | 39 passed.                                       |
| libs/partners/mistralai       |        65% |           389 |           212 | 38 passed, 1 skipped.                            |
| libs/partners/nomic           |        78% |            21 |             6 | 3 passed.                                        |
| libs/partners/ollama          |        78% |           635 |           184 | 92 passed, 2 skipped.                            |
| libs/partners/openai          |        73% |         2,326 |           873 | 384 passed, 5 xpassed.                           |
| libs/partners/openrouter      |        94% |           578 |            36 | 220 passed.                                      |
| libs/partners/perplexity      |        69% |           352 |           161 | 52 passed, 1 skipped.                            |
| libs/partners/qdrant          |        34% |           257 |           495 | 2 passed (most tests require a running Qdrant).  |
| libs/partners/xai             |        69% |            83 |            37 | 27 passed.                                       |

## Notes on the run

- Partner unit-test suites that depend on a live service (Chroma DB, Qdrant,
  Ollama daemon, etc.) only exercise the lightweight import / config paths
  without that service. Those baselines should be re-collected once we have a
  way to spin the services up in CI; treat the numbers above as a floor.
- The full `make test` for `libs/langchain_v1` requires Postgres + Redis via
  Docker Compose. The 91% above is from `pytest tests/unit_tests/
  --disable-socket --allow-unix-socket`, which is the same flow the new
  `coverage.yml` workflow runs.

## Trajectory

We are setting CI `--fail-under` to the **current baseline rounded down to the
nearest 5%** so this PR is purely additive (no existing CI run regresses), and
ratcheting from there.

| Package           | Baseline | Initial `fail_under` | Long-term target |
|-------------------|---------:|---------------------:|-----------------:|
| libs/core         |      83% |                  80% |              90% |
| libs/langchain_v1 |      91% |                  90% |              95% |
| libs/langchain    |      74% |                  70% |              80% |
| libs/text-splitters |    79% |                  75% |              85% |

For partner packages the threshold stays at `0%` (advisory only) until we have
a way to run partner-specific tests against mock servers in CI. Their numbers
above are the starting line so we can spot regressions during code review even
before a hard threshold is enforced.

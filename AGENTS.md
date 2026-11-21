# Repository Guidelines

## Project Structure & Module Organization

TileLens is a Python package with a TypeScript web UI. Core code lives in `tilelens/`: `core/` handles tracing and client lifecycle, `core/frontend/` owns DSL operation adapters, runtime patching, and frontend-local AST rewrites, `core/simulation/` contains NKI simulation runtimes, `clients/` contains visualizer/profiler/sanitizer logic, and `visualizer/` covers visualizer APIs. Web UI source is in `frontend/`; built assets are copied into `tilelens/static/` and `tilelens/templates/`. Tests live in `tests/unit/`, `tests/end_to_end/`, `tests/frontend/`, and `tests/nki/`. Examples are in `examples/`; site docs are in `docs/`.

## Build, Test, and Development Commands

- `uv sync --extra test`: install Python test dependencies.
- `uv sync --extra nki --extra test`: install NKI plus test dependencies; specify all extras in one command.
- `uv run pytest tests/`: run the default Python suite, excluding tests marked `nki`.
- `uv run pytest tests/ -m ""`: run all Python tests, including NKI tests when dependencies are installed.
- Without a GPU, the profiler tests in `tests/unit/test_multithreading.py` cannot compile. Add `--ignore=tests/unit/test_multithreading.py` to the commands above and run that file with `TRITON_INTERPRET=1 uv run pytest tests/unit/test_multithreading.py`.
- `npm install`: install web UI tooling.
- `npm run build:frontend`: compile TypeScript and copy web UI package assets.
- `npm run test:frontend`: build the web UI and run Node-based web UI tests.

## Coding Style & Naming Conventions

Use Python 3.12+ and TypeScript modules. Follow file-local style: 4-space indentation for Python and descriptive snake_case names for functions, variables, and test helpers. Python tests are discovered from `*.py`, classes ending in `Test`, and functions named `test_*`. Ruff is configured in `pyproject.toml` with `E731` ignored; run pre-commit when available. Keep generated package assets in sync when editing `frontend/`.

## Testing Guidelines

Prefer focused tests near the behavior being changed: unit tests in `tests/unit/`, kernel or CLI-visible behavior in `tests/end_to_end/`, and web UI tests in `tests/frontend/*.mjs`. Default pytest skips NKI tests; use `-m nki` for only NKI tests or `-m ""` for the full suite. For sanitizer work, cover records, reports, CLI behavior, and launch lifecycle where applicable.

## Commit & Pull Request Guidelines

Recent commits use concise imperative summaries, often with prefixes such as `[FIX]`, `[FEAT]`, or `[REFACTOR]`, and PR numbers added by GitHub, for example `[FIX] Handle stride-0 expanded tensors in sanitizer (#327)`. Keep commits scoped to one logical change. Pull requests should explain the user-visible change, list test commands run, link related issues, and include screenshots or trace examples when web UI or visualizer output changes.

PRs are squash-merged, so keep a stack of PRs up to date with merges, not rebase + force-push:

- Push new work as new commits on the branch it belongs to.
- Then update every PR above it, bottom-up, one at a time. Use the PR page's **Update branch** button (Update with merge commit) or `gh pr update-branch <number>` without `--rebase`.
- When the bottom PR is squash-merged, GitHub rebases the next PR onto `main` and force-updates its branch, dropping its merge commits; the PRs above it are left alone. Reset your local branch to the rebased one instead of pushing the old one. In the PR above it, merge the pre-rebase tip first if it is not merged yet; then, once `git diff <old-tip> <rebased-tip>` is empty, run `git merge -s ours <rebased-tip>`. A plain merge of the rebased tip conflicts even where only the lower PR changed code, because the rebase cut their shared history.
- Take a lower PR's changes only through merges. Never add a commit that sets files back to the base branch's version: that rebase replays it as a revert of the merged change.
- Do not use `gh stack sync`, which rebases and force-pushes, or `gh stack push`, which force-pushes.

## Security & Configuration Tips

Do not commit local traces, credentials, or generated debug artifacts. Runtime behavior is controlled by environment variables such as `TILELENS_VERBOSE`, `TILELENS_PORT`, `ENABLE_SANITIZER`, `ENABLE_PROFILER`, and `SANITIZER_ENABLE_FAKE_TENSOR`; document any new variables in `README.md`.

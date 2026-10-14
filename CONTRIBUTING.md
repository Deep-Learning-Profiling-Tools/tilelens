# Contributing to TileLens

## Quick Start

```bash
# Clone repo
git clone https://github.com/Deep-Learning-Profiling-Tools/tilelens.git
cd tilelens

# Setup dev environment and install dependencies
uv sync --extra test  # for NKI work: uv sync --python 3.12 --extra nki --extra test
pre-commit install

# (code up the PR)

# Test
uv run pytest tests/ -m ""      # run all tests
npm run build:frontend           # build the web UI (if working on the visualizer)
npm run test:frontend            # test the web UI (if working on the visualizer)
```

## Project Structure

```
frontend/             # TypeScript web UI source code
examples/             # Entry points for new users to try out tilelens functionality
docs/                 # tilelens website
tilelens/
  core/frontend/      # DSL operation adapters and runtime patching
  core/simulation/    # CPU simulation runtimes for Gluon and NKI
  clients/            # Tracer (visualizer), profiler, sanitizer, and race detector clients
  visualizer/         # Python interface for the visualizer
  utils/              # Miscellaneous utility functions
tests/
  unit/               # Unit tests
  end_to_end/         # End-to-end tests
  frontend/           # Web UI tests
  nki/                # NKI-related tests
```

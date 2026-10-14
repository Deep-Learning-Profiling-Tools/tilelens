<a name="readme-top"></a>
# TileLens: A Visualization Toolkit for Tile-Based Languages
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="https://raw.githubusercontent.com/Deep-Learning-Profiling-Tools/tilelens/main/docs/logo.png" alt="TileLens logo" width="320" height="320">
</div>
<br/>

Welcome to TileLens, a visualization, profiling, and debugging toolkit for deep learning kernels written in tile-based languages. TileLens currently supports Triton, Gluon, and AWS Neuron NKI, and aims to make tile-based kernel programming more intuitive.

Visit our [site](https://deep-learning-profiling-tools.github.io/tilelens/) to see our tool in action!

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-of-tilelens">Installation of TileLens</a></li>
        <li><a href="#transitioning-from-triton-viz-to-tilelens">Transitioning from Triton-Viz to TileLens</a></li>
        <li><a href="#web-ui-build">Web UI Build</a></li>
        <li><a href="#optional-enable-nki-support">Optional: Enable NKI Support</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#working-with-examples">Working with Examples</a></li>
    <li><a href="#dsl-frontends">DSL Frontends</a></li>
    <li>
      <a href="#analysis-clients">Analysis Clients</a>
      <ul>
        <li><a href="#save-and-load-traces">Save and load traces</a></li>
      </ul>
    </li>
    <li><a href="#environment-variables">Environment Variables</a></li>
    <li><a href="#more-puzzles">More Puzzles</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#publication">Publication</a></li>
  </ol>
</details>

## About

TileLens helps developers inspect kernels written in tile-based languages with visualization, profiling, and memory-safety (out-of-bounds) analysis tools. Tool support varies by language; see [Analysis Clients](#analysis-clients). Kernels run on the CPU: Triton kernels through Triton's interpreter, Gluon kernels through TileLens's NumPy-backed Gluon simulator, and NKI kernels through TileLens's NKI simulators. Basic debugging workflows therefore need neither a GPU nor a Trainium device.


## Getting Started

### Prerequisites
- Linux on x86_64 or aarch64. TileLens requires Triton for every language, including NKI, and Triton only publishes Linux wheels. On Windows, install and run TileLens inside Windows Subsystem for Linux (WSL). macOS is not supported natively.
- Python 3.12-3.14. 3.12 is the minimum, and 3.14 is the newest version Triton 3.8 publishes wheels for. NKI support needs Python 3.12 on Linux x86_64.
- PyTorch. TileLens imports `torch` at startup but does not install it, so install it first. A CPU-only build is enough. If you use the default PyPI build, use torch 2.14 or newer: earlier releases pin a Triton older than the 3.8 that TileLens requires.
- A browser with WebGL enabled to view the visualizer.
- For source installs: [uv](https://docs.astral.sh/uv/). Node.js and npm are only needed to rebuild the web UI.


### Installation of TileLens

Install TileLens from PyPI:

```sh
pip install "torch>=2.14"  # or a CPU-only build; see Prerequisites
pip install tilelens
```

If you want to run examples from this repo, contribute (see [CONTRIBUTING.md](https://github.com/Deep-Learning-Profiling-Tools/tilelens/blob/main/CONTRIBUTING.md)), or build the web UI, install from source instead:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/tilelens.git
cd tilelens
uv sync --extra test  # the test extra also installs PyTorch, which TileLens and the examples need
```

Then run commands through `uv run` (for example `uv run python examples/visualizer/matmul.py` or `uv run tile-sanitizer ...`), or activate `.venv` first.

### Transitioning from Triton-Viz to TileLens

The GitHub repo and PyPI package are now named `tilelens`. Use `import tilelens`
in new code. Old `triton_viz` imports, including submodule imports, still work.

If you already have `triton-viz` installed, uninstall it before installing
TileLens. The two packages share files, so keeping both installed can break
imports and CLI commands:

```sh
pip uninstall -y triton-viz
pip install tilelens
```

For source installs, use `pip install .` after uninstalling the old package.
If you already installed both, uninstall `triton-viz` first, then run
`pip install --force-reinstall --no-deps tilelens` (or
`pip install --force-reinstall --no-deps .` from this repo). This restores the
shared files without reinstalling or upgrading dependencies such as Triton.
Restart Python or your notebook kernel after upgrading.

Existing `.tvz` traces can still be loaded with `tilelens.load(...)`.
Traces saved by TileLens cannot be loaded by older Triton-Viz versions.

CLI commands are `tile-sanitizer`, `tile-profiler`, `tile-race`, and `tile-visualizer`.
The old `triton-sanitizer`, `triton-profiler`, `triton-race-detector`, and
`triton-visualizer` commands still work.

The old environment variable names `TRITON_VIZ_VERBOSE`, `TRITON_VIZ_NUM_SMS`,
and `TRITON_VIZ_PORT` are still read; see [Environment Variables](#environment-variables).

### Web UI Build

Prebuilt web UI assets are committed under `tilelens/static` and
`tilelens/templates` and ship in the PyPI package, so you do not need npm to run
the visualizer from either a PyPI or a source install. To modify the web UI
(TypeScript sources in `frontend/`), install Node.js and npm, rebuild, and
commit the regenerated files in `tilelens/static` and `tilelens/templates`
together with your source changes:

```sh
npm install
npm run build:frontend
```

### Optional: Enable NKI Support

NKI support requires Linux x86_64 and Python 3.12; on other Python
versions the `nki` extra installs nothing. The extra provides both NKI
frontends: `neuronx-cc` for `frontend="nki"` and the standalone `nki` package
for `frontend="nki_beta2"`. Kernels run in TileLens's CPU simulators, so no
Trainium hardware is needed.

For PyPI installs, install with the `nki` extra and AWS Neuron repository:

```sh
pip install "tilelens[nki]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

For source installs, pin a supported interpreter:

```sh
uv sync --python 3.12 --extra nki --extra test
```

`uv sync` installs exactly the extras you name and removes the rest, so list
every extra _in one command_. Running these one after the other leaves you
without NKI:

```sh
uv sync --extra nki
uv sync --extra test  # removes the NKI packages installed above
```

### Testing

Install the test extra first: `uv sync --extra test` (or `uv sync --python 3.12 --extra nki --extra test` for NKI).

* Default suite (Triton and Gluon; NKI tests skipped): `uv run pytest tests/` (add `-n auto` to run in parallel).
* NKI-only tests (requires the `nki` extra): `uv run pytest tests/ -m nki`.
* All marked tests (Triton, Gluon, and NKI): `uv run pytest tests/ -m ""`.
* `triton_kernels` sanitizer tests are excluded by default. Install the `triton_kernels` that matches your Triton release (for Triton 3.8.0: `uv pip install "git+https://github.com/triton-lang/triton.git@v3.8.0#subdirectory=python/triton_kernels"`), then run `uv run pytest tests/end_to_end/test_triton_kernels.py`. `--triton-kernels-device=cpu|cuda` selects the device; the swiglu cases read the CUDA GPU's SM count, so they fail on machines without a CUDA GPU even with `cpu`.
* Visualizer web UI tests: `npm install && npm run test:frontend`.

## Working with Examples

[`examples/`](https://github.com/Deep-Learning-Profiling-Tools/tilelens/tree/main/examples) has runnable scripts for each language and client. The
examples below run on the CPU, so no GPU is needed. They import torch, so
install from source with `uv sync --extra test` (for the NKI examples, use
`uv sync --python 3.12 --extra nki --extra test` instead):

```sh
uv run python examples/visualizer/matmul.py        # Triton + visualizer; serves the UI at http://localhost:5001 (Ctrl+C to stop)
uv run python examples/sanitizer/gluon_tma_oob.py  # Gluon + sanitizer; prints a TMA out-of-bounds report and exits with status 1
uv run python examples/nki/matmul.py               # NKI (needs the nki extra); serves the UI at :5001
uv run python examples/nki_beta2/add.py            # NKI Beta 2 (needs the nki extra); serves the UI at :5001
```

Use the `@tilelens.trace(...)` decorator when writing or modifying a kernel.
For non-Triton kernels, pass `frontend="gluon"`, `"nki"`, or `"nki_beta2"` (see
[DSL Frontends](#dsl-frontends)). A Triton example:

```py
import torch
import triton
import triton.language as tl
import tilelens


# Or "tracer" (the default; feeds the visualizer) or "profiler" (see below)
@tilelens.trace("sanitizer")
@triton.jit
def copy_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets))


x = torch.arange(16, dtype=torch.float32)
out = torch.empty_like(x)
copy_kernel[(1,)](x, out, BLOCK=16)  # runs on the CPU, no GPU needed
```

The profiler's AMD `buffer_load` check compiles the kernel for the local GPU, so on
a machine without a GPU set `PROFILER_DISABLE_BUFFER_LOAD_CHECK=1` before using
`"profiler"` or `tile-profiler` (see [Analysis Clients](#analysis-clients)).

Pass a client object instead of a name to configure it, for example
`tilelens.trace(Tracer(grid_idx=(0, 0, 0)))` to record only program `(0, 0, 0)`
(`from tilelens.clients import Tracer`). Stack `@tilelens.trace(...)`
decorators to run several clients on one kernel, and call `tilelens.clear()` to
drop all recorded launches.

Use the CLI wrappers to run an existing script without editing it.
`tile-sanitizer` and `tile-profiler` patch Triton `@triton.jit` kernels
(including ones wrapped in `@triton.autotune`) and Gluon `@gluon.jit` kernels
before running the script. The profiler supports Triton kernels only; a Gluon
kernel launched under `tile-profiler` currently fails with an `InterpreterError`,
so use `tile-profiler` only on scripts that launch Triton kernels. NKI kernels
are not patched, and the sanitizer and profiler do not support NKI; to visualize
NKI kernels, use `tilelens.trace("tracer", frontend="nki")` (or `"nki_beta2"`).
Do not run the wrappers on scripts whose Triton or Gluon kernels already use
`@tilelens.trace(...)`; that raises a `RuntimeError`.

```sh
tile-sanitizer examples/sanitizer/oob_cli.py       # arguments after the script are passed to it
tile-profiler examples/profiler/load_store_cli.py  # without a GPU: PROFILER_DISABLE_BUFFER_LOAD_CHECK=1 tile-profiler ...
tile-sanitizer pytest tests/test_my_kernels.py -x  # wrap a pytest run (needs pytest)
```

The first argument can be a script path, `python <script.py> [args...]`, or
`pytest [args...]`; other commands, including `python3`, are rejected.

For the visualizer, trace the kernel with the `"tracer"` client (the default
when no client is given), run it, then open the UI or save the trace:

```py
@tilelens.trace("tracer")
@triton.jit
def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...


kernel[grid](x, n_elements, BLOCK_SIZE=1024)
# No public link; serves http://localhost:5001 and blocks until Ctrl+C
tilelens.launch(share=False)
# Or: tilelens.save("trace.tvz"), then later `tile-visualizer trace.tvz`
```

`tilelens.launch()` defaults to `share=True`, which publishes the UI, including
kernel source and tensor data, through a public Cloudflare URL. `share=False`
creates no public link, but the Flask development server (debug mode) still
listens on all network interfaces (`0.0.0.0`), so other machines that can reach
this host on that port can open the UI; block the port with a firewall on shared
or untrusted networks.

## DSL Frontends

Choose the language with `tilelens.trace(client, frontend=...)`. Put
`@tilelens.trace(...)` outermost, above the language's own `jit`/`autotune`
decorators, and launch with `kernel[grid](...)`.

| `frontend=` | Language | Extra install |
|---|---|---|
| `"triton"` (default) | Triton (`@triton.jit`) | none |
| `"gluon"` | Triton Gluon (`@gluon.jit`) | none (part of Triton) |
| `"nki"` | AWS NKI (`neuronxcc.nki`, from neuronx-cc) | `tilelens[nki]`; Python 3.12, Linux x86_64 |
| `"nki_beta2"` | AWS NKI Beta 2 (standalone `nki` package: `nki.isa`, `nki.language`) | same as `"nki"` |

The decorator does not auto-detect Gluon kernels, so pass `frontend="gluon"`
explicitly. The CLI wrappers do this for you.

```py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import tilelens


@tilelens.trace("tracer", frontend="gluon")
@gluon.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK: gl.constexpr, layout: gl.constexpr):
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    gl.store(out_ptr + offs, gl.load(in_ptr + offs, mask=mask, other=0.0), mask=mask)


x = torch.arange(40, dtype=torch.float32)
out = torch.empty_like(x)
copy_kernel[(1,)](
    x, out, x.numel(), 64, gl.BlockedLayout([1], [32], [1], [0]), num_warps=1
)
```

Pass NKI kernels as plain Python functions, not `nki.jit` objects. `"nki"`
kernels take NumPy arrays; `"nki_beta2"` kernels take NumPy arrays or CPU torch
tensors:

```py
# Use frontend="nki" for neuronxcc.nki kernels
traced = tilelens.trace("tracer", frontend="nki_beta2")(nki_kernel)
traced[(1,)](a, b, result)
```

For `"nki_beta2"`, TileLens first runs the NKI compiler's tracer to catch
semantic errors. Pass `pre_trace=False` at launch to skip it; this is required
if the kernel uses arbitrary Python, such as imports inside the kernel.

See `examples/sanitizer/gluon_tma_oob.py`, `examples/nki/`, and
`examples/nki_beta2/` for complete scripts. DSL adapters live in
`tilelens/core/frontend/`, and the CPU simulators for Gluon and NKI live in
`tilelens/core/simulation/`.

## Analysis Clients

Analyze kernels across visualization, profiling, and sanitization with a single line of code.

| Client | Triton | Gluon | NKI / NKI Beta 2 |
|---|---|---|---|
| `"tracer"` (visualizer) | yes | yes | yes |
| `"sanitizer"` | yes | yes | no |
| `"profiler"` | yes | no | no |

- Visualizer (`"tracer"` client): records loads, stores, dot/matmul operations, and NKI Beta 2 DMA/tensor-copy transfers (`dma_copy`, `tensor_copy`), and renders them in a 3D web UI for tensors of any rank. Tensors with more than three dimensions are drawn as nested 3D grids, and you can slice, reorder, or hide axes in the UI.
- Profiler (`"profiler"`): lists each `for` loop with its range kind (`python_range`, `tl_range`, or `tl_static_range`) and trip count, reports load/store mask ratios, and records load/store byte counts. To keep overhead low, by default it runs one randomly sampled program per launch and skips the real loads, stores, and `tl.dot` math, so kernel outputs are not computed (see [Environment Variables](#environment-variables); `Profiler(k=N)` samples `N` programs). Programs it skips are skipped for every client stacked on the same kernel. It also checks whether the compiled kernel uses AMD `buffer_load` instructions. That check compiles the kernel for the local GPU, so it is only meaningful on AMD GPUs and fails on machines without a GPU; set `PROFILER_DISABLE_BUFFER_LOAD_CHECK=1` on non-AMD or GPU-less machines.
- Sanitizer (`"sanitizer"`): symbolically checks loads and stores, including Gluon TMA descriptor accesses, for out-of-bounds errors. By default it prints a report at the first violation (tensor metadata, source context, call stack, and symbolic expression tree) and exits with status 1. To collect violations instead, trace with `san = Sanitizer(abort_on_error=False)` (`from tilelens.clients import Sanitizer`) and read the `san.records` list. Loads and stores are evaluated symbolically, so Triton and Gluon kernel outputs are not written. Optional fake-tensor mode (`SANITIZER_ENABLE_FAKE_TENSOR=1`, Triton only) avoids copying tensor data to host memory.

### Save and load traces

```py
import tilelens

tilelens.save("trace.tvz")
tilelens.load(
    "trace.tvz"
)  # automatically clears out existing records, use kwarg "append=True" to prevent this
tilelens.launch(share=False)
```

To reopen a saved trace, run `tile-visualizer trace.tvz [--port PORT] [--share]`.
It serves the UI on port 5001 by default until stopped; as with
`launch(share=False)`, there is no public link, but the server listens on all
network interfaces. `--share` also opens a public Cloudflare link (default port
8000). The archive is a zip file
containing `manifest.json` plus `tensors.npz`, which holds full copies of the
traced tensors, and `tilelens.load(...)` restores the normal trace state for
existing consumers. Saving bfloat16 tensors is not supported yet.


## Environment Variables

TileLens uses a small set of environment variables to configure runtime behavior. Unless noted, boolean flags are enabled only when set to `1`.

TileLens reads these variables once, when `tilelens` is first imported, so set
them before the import. `TILELENS_PORT` is the exception: `launch()` reads it
each time it runs. To change a setting after import, assign the attribute on
`tilelens.config` (for example `tilelens.config.num_sms = 4`; attribute names
are listed in `tilelens/core/config.py`, e.g. `SANITIZER_ENABLE_FAKE_TENSOR`
maps to `virtual_memory`), or update `os.environ` and call
`tilelens.config.reset()`. Do this before the kernel is decorated:
`ENABLE_SANITIZER` and the `PROFILER_*` flags are captured when
`@tilelens.trace(...)` creates the client.

The old names `TRITON_VIZ_VERBOSE`, `TRITON_VIZ_NUM_SMS`, and `TRITON_VIZ_PORT`
still work. If both names are set, TileLens uses the `TILELENS_*` value.

- `TILELENS_VERBOSE` (default: `0`): enable verbose logging and extra debug output.
- `TILELENS_NUM_SMS` (default: `1`, min `1`): number of worker threads that run Triton program instances concurrently on Triton's CPU interpreter, emulating multiple SMs. Only the Triton frontend uses it; Gluon and NKI kernels run program instances one at a time.
- `TILELENS_PORT` (default: `8000` when sharing through a Cloudflare tunnel, otherwise `5001`): port for the visualizer's Flask server when no `port=` argument or `--port` flag is given. `tilelens.launch()` shares by default (`share=True`); `tile-visualizer` shares only with `--share`. When sharing, cloudflared's metrics endpoint also binds the next port.
- `ENABLE_SANITIZER` (default: `1`): enable the sanitizer. When `0`, kernels traced with the sanitizer, including under `tile-sanitizer`, run untraced.
- `ENABLE_TIMING` (default: `0`): print the wall-clock time of each traced Triton kernel launch (Triton frontend only).
- `SANITIZER_ENABLE_FAKE_TENSOR` (default: `0`, Triton only): pass tensor arguments as metadata-only fake tensors that keep the original data pointers, so tensor data is not copied to host memory and back. Intended for sanitizer runs. It applies to every traced Triton launch regardless of client, and any real memory access then dereferences the original pointer, which crashes for GPU tensors: tracer loads and stores, profiler loads and stores with `PROFILER_ENABLE_LOAD_STORE_SKIPPING=0`, and sanitizer loads whose values feed later addresses (e.g. indirect indexing).
- `SANITIZER_REPORT_MAX_SEGMENTS` (default: `8`, min `2`): maximum number of address segments listed in an out-of-bounds report before it is shortened to head/tail entries (display only).
- `SYMBOLIC_PER_ELEMENT_WARN_THRESHOLD` (default: `8192`): the sanitizer warns when a non-contiguous tensor argument with non-unit inner stride has more elements than this, because per-element address enumeration can slow the solver. Set to `0` to disable the warning.
- `PROFILER_ENABLE_LOAD_STORE_SKIPPING` (default: `1`): during profiling, do not execute loads, stores, or `tl.dot` (loads and dots return zeros, stores are dropped) to reduce overhead. Byte counts and mask statistics are still collected, but kernel outputs are not computed; set to `0` if the kernel needs real data (e.g. data-dependent control flow).
- `PROFILER_ENABLE_BLOCK_SAMPLING` (default: `1`): execute only a random sample of program instances per launch to reduce profiling overhead. The default sample is one instance, which is what `tile-profiler` always uses; from Python, trace with `tilelens.clients.Profiler(k=N)` to sample more. Skipped instances are skipped for every client stacked on the same kernel. Set to `0` to profile every program instance.
- `PROFILER_DISABLE_BUFFER_LOAD_CHECK` (default: `0`): disable the AMD-only `buffer_load` check, which compiles the kernel for the local GPU. Set to `1` on non-AMD or GPU-less machines.

## More Puzzles

If you're interested in fun puzzles for learning Triton, check out [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles), which is built on Triton-Viz, the predecessor of TileLens.

## License

TileLens is licensed under the MIT License. See the [LICENSE](https://github.com/Deep-Learning-Profiling-Tools/tilelens/blob/main/LICENSE) for details.

## Publication
If you find this repo useful for your research, please cite our paper:

```
@inproceedings{ramesh2025tritonviz,
  author={Ramesh, Tejas and Rush, Alexander and Liu, Xu and Yin, Binqian and Zhou, Keren and Jiao, Shuyin},
  title={Triton-Viz: Visualizing GPU Programming in AI Courses},
  booktitle = {Proceedings of the 56th ACM Technical Symposium on Computer Science Education (SIGCSE TS '25)},
  numpages = {7},
  location = {Pittsburgh, Pennsylvania, United States},
  series = {SIGCSE TS '25}
}

@inproceedings{wu2026tritonsanitizer,
  author    = {Wu, Hao and Zhao, Qidong and Chen, Songqing and Chen, Yang and Hao, Yueming and Liu, Tony C. W. and Chen, Sijia and Aziz, Adnan and Zhou, Keren},
  title     = {Triton-Sanitizer: A Fast and Device-Agnostic Memory Sanitizer for Triton with Rich Diagnostic Context},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Pittsburgh, PA, USA},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  series    = {ASPLOS '26},
  keywords  = {GPU, Debugging, Symbolic Execution, Memory Safety, Triton, Memory Access Errors}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="readme-top"></a>
# TileLens: A Visualization Toolkit for programming with Triton
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="docs/logo.png" alt="Logo" width="320" height="320">
</div>
<br/>

Welcome to TileLens, a visualization and profiling toolkit designed for deep learning applications. Built with the intention of making kernel programming in tile-based DSLs like Triton more intuitive.

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
      </ul>
    </li>
    <li>
      <a href="#working-with-examples">Working with examples</a>
    </li>
    <li><a href="#dsl-frontends">DSL frontends</a></li>
    <li><a href="#analysis-clients">Analysis clients</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About

TileLens helps developers inspect Triton kernels with visualization, profiling, and memory-safety analysis tools. It can run many examples through Triton's interpreter, so GPU access is not required for basic debugging workflows.


## Getting Started

### Prerequisites
- Python >= 3.10


### Installation of TileLens

> **Windows Note:** TileLens depends on Triton, which can only be installed on Windows Subsystem for Linux (WSL). Once installed, follow below instructions in WSL.

Install TileLens from PyPI:

```sh
pip install tilelens
```

If you want to run examples from this repo, contribute, or build the web UI, install from source instead:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/tilelens.git
cd tilelens
uv sync # or "uv sync --extra test" if you're running tests
```

### Transitioning from Triton-viz to TileLens

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

### Web UI Build

The PyPI package ships with prebuilt web UI assets in `tilelens/static`, so
you do not need npm to run the visualizer. If you want to modify the web UI,
rebuild the TS sources:

```sh
npm install
npm run build:frontend
```

### Optional: Enable NKI Support

For PyPI installs, install with the `nki` extra and AWS Neuron repository:

```sh
pip install "tilelens[nki]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

For source installs:

```sh
uv sync --extra nki # or "uv sync --extra nki --extra test" if also running NKI-related tests
```

Note that you need to specify all features that you want _in one statement_ when using `uv sync`, i.e. if you want both NKI and testing support, you must run `uv sync --extra nki --extra test`. The below statements are wrong and will remove the NKI install when installing test packages:
```
uv sync --extra nki # NKI support but no testing
uv sync --extra test # tests but no NKI support
```

### Testing
* To run core TileLens tests, run `pytest tests/`.
* (if NKI installed) To run NKI-specific tests, run `pytest tests/ -m nki`.
* To run all tests (Triton + NKI), run `pytest tests/ -m ""`.
* The IR-mode tests (the compiled sanitizer, below, and its IR layer) need
  Triton 3.8, the one release IR mode runs on; on another release they fail.
* To run visualizer web UI tests, run `npm run test:frontend`.

## Working with Examples

Run an example directly with Python:

```sh
python examples/visualizer/matmul.py
```

Use the decorator API when writing or modifying a Triton kernel:

```py
import triton
import triton.language as tl
import tilelens


@tilelens.trace("sanitizer")  # also supports "tracer" and "profiler"
@triton.jit
def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, values)
```

Use the CLI wrappers to run an existing Python script without editing it. These
wrappers patch plain `@triton.jit` kernels, so use them with scripts that do not
already apply `@tilelens.trace(...)`.

```sh
tile-sanitizer examples/sanitizer/oob_cli.py
tile-profiler examples/profiler/load_store_cli.py
tile-visualizer trace.tvz
```

For visualizer workflows, save a trace and launch the UI from Python:

```py
import tilelens

tilelens.save("trace.tvz")
tilelens.launch()
```

## DSL Frontends

Triton is the default DSL frontend. NKI support is optional and selected with
the `frontend` argument:

```py
tilelens.trace("tracer")  # Triton
tilelens.trace("tracer", frontend="nki")  # NKI
tilelens.trace("tracer", frontend="nki_beta2")  # NKI Beta 2
```

The runtime integration code lives under `tilelens/core/frontend/`. NKI
simulation runtimes live under `tilelens/core/simulation/`.

## Analysis Clients

Analyze kernels across visualization, profiling, and sanitization with a single line of code.

- Visualizer: currently supports load, store, and matmul operations for 1/2/3D tensors (more operations and dimensions coming soon).
- Profiler: flags non-unrolled loops, inefficient mask usage, and missing buffer_load optimizations while tracking load/store byte counts with low-overhead sampling.
- Sanitizer: symbolically checks tensor memory accesses for out-of-bounds errors and emits reports with tensor metadata, call stack, and expression trees; optional fake-memory storage avoids real reads.

### Compiled sanitizer

`Sanitizer(compile=True)` checks each launch against the kernel Triton compiles
for it (its TTIR) instead of interpreting it: out-of-bounds accesses (against the
tensor's view, strides included), integer-width overflows in address, mask and
branch arithmetic, and divisions by zero, each with a witness (program ids, lanes,
loop iteration). The kernel is compiled on the host, exactly as the JIT would
compile the launch but only through the TTIR stage, so **no GPU is needed**: CPU
tensors work, and so does a machine without a driver. From the CLI, give the
flag before the script name (the legacy `triton-sanitizer` alias takes it too):

```sh
tile-sanitizer --compile my_script.py --my-script-flag
```

```py
from tilelens.clients import Sanitizer


@tilelens.trace(Sanitizer(compile=True))
@triton.jit
def kernel(x_ptr, n, BLOCK: tl.constexpr):
    ...
```

- Kernels are compiled and checked, **not run**: their outputs are never written,
  so a script that checks its own results fails under `--compile`, and a launch
  whose arguments the script computes from an earlier kernel's output is checked
  with those unwritten values.
- Kernels are compiled for a fixed target, `cuda:89` (sm89, e.g. RTX 4090) by
  default, so a verdict does not depend on the machine it was computed on.
  Choose another with `Sanitizer(compile=True, target="cuda:90")` (or
  `"hip:gfx942"`, or a Triton `GPUTarget`), or for every client that names none
  with the environment variable `TILELENS_IR_TARGET=cuda:90` (e.g. for
  `tile-sanitizer --compile`). The kernel's
  own target queries (`tl.target_info.is_cuda()`, `cuda_capability_geq()`,
  `is_hip()`) answer for that target too, and `TRITON_OVERRIDE_ARCH` does not
  apply: the target is the one named. The TTIR can differ between targets (e.g.
  tensor descriptors, target-dependent branches), and a verdict holds for the
  target it was checked for.
- Targets also differ in what compiles at all: `fp8e4nv` (`torch.float8_e4m3fn`,
  `tl.float8e4nv`) needs `cuda:89` or later, `num_ctas > 1` and 16-bit tensor
  descriptor atomic min/max need `cuda:90`. A kernel or autotune config that fails
  to compile for the target never stops the script (it was not going to run
  anyway): it is reported `unsupported`, kind `compile-failed`, naming the target
  and how to choose another, since it may run on a GPU of another kind unchecked.
  Name a target it compiles for to check it. Only a failure no target compiles
  past (a failing `tl.static_assert`, or a Python construct Triton never compiles,
  unless the kernel asked Triton's driver anything first, e.g. through
  `tl.target_info` or a device query it catches, or an earlier compile of the
  kernel for the target did) is just a note in the verdict: that config never
  launches (the autotuner skips it too). A target answer the kernel's own code
  keeps from outside the check (another trace or target, the untraced program)
  and never asks for again cannot be seen. A launch none of whose configs
  compiled is `unsupported` (`compile-failed`), and its notes are printed with
  it. Each report names where the kernel failed (`file:line`) and the innermost
  error in one line.
- A call that does not match the kernel's signature (a missing, extra or
  misnamed argument), or that Triton cannot key (e.g. an unhashable constexpr
  value), is a bug in the call, not a compile failure: it raises the very
  `TypeError` the untraced call raises, on any GPU, and the script stops there
  (under `tile-sanitizer --compile` with a traceback and exit status 1). So does
  a trace that also interprets the kernel (e.g. with the `Tracer`), before the
  interpreter runs, which would fail on the call too. An autotuned call that
  passes an autotuned meta-parameter itself raises the autotuner's own
  `Conflicting meta-parameters` `ValueError`. A keyword that names no parameter
  is a compile option, which the target may not know (e.g. `waves_per_eu`, a
  HIP option, under a CUDA target): that is the target's `compile-failed`, and
  the report names the keyword (misspelled, it fails on every GPU untraced too).
  All of this holds on Triton 3.8 (below): on another release nothing is
  compiled, so nothing binds the call, and the launch is `unsupported`
  (`host-compile-unavailable`), a call that does not bind included (a trace
  that also interprets the kernel raises the interpreter's own error for it).
- Each launch gets an `IRVerdict` in its records: `ok` is a proof for that
  launch's scalar arguments, grid and tensors (`scope="launch"`); `violations`
  comes with the findings; `unsupported` names what was not checked (e.g. a
  data-dependent address, a construct the TTIR reader does not model, a Z3
  query that timed out, which can depend on the machine's load, a config that
  failed to compile for the target, `compile-failed`, or a compile the host
  cannot run, `host-compile-unavailable`, e.g. under `TRITON_KERNEL_OVERRIDE`,
  `USE_IR_LOC` or a custom pipeline, which change the TTIR through stages the
  host compile does not run). An autotuned
  launch checks every config, each with its own arguments and grid and with a
  `ConfigVerdict` of its own, also when several configs compile to one kernel.
- With the default `abort_on_error=True` the findings are printed and the
  process exits with status 1; `ENABLE_SANITIZER=0` leaves kernels untraced.
- Requires Triton 3.8; on another release every launch is `unsupported`
  (`host-compile-unavailable`), nothing compiled, and the program goes on.

### Save and load traces

```py
import tilelens

tilelens.save("trace.tvz")
tilelens.load(
    "trace.tvz"
)  # automatically clears out existing records, use kwarg "append=True" to prevent this
tilelens.launch()
```

CLI: `tile-visualizer trace.tvz`. The archive is a zip file containing `manifest.json` plus `tensors.npz`, and `tilelens.load(...)` restores the normal trace state for existing consumers.


### Environment variables

TileLens uses a small set of environment variables to configure runtime behavior. Unless noted, boolean flags are enabled only when set to `1`.

The old names `TRITON_VIZ_VERBOSE`, `TRITON_VIZ_NUM_SMS`, and `TRITON_VIZ_PORT`
still work. If both names are set, TileLens uses the `TILELENS_*` value.

- `TILELENS_VERBOSE` (default: `0`): enable verbose logging and extra debug output.
- `TILELENS_NUM_SMS` (default: `1`): number of concurrent SMs to emulate for the CPU interpreter (min 1).
- `TILELENS_PORT` (default: `8000` with `share=True`, `5001` with `share=False`): port for the Flask server.
- `ENABLE_SANITIZER` (default: `1`): enable the sanitizer pipeline that checks memory accesses.
- `ENABLE_PROFILER` (default: `1`): enable the profiler pipeline that collects performance data.
- `ENABLE_TIMING` (default: `0`): collect timing data during execution.
- `REPORT_GRID_EXECUTION_PROGRESS` (default: `0`): report per-program block execution progress in the interpreter.
- `SANITIZER_ENABLE_FAKE_TENSOR` (default: `0`): use fake tensor storage for sanitizer runs to avoid real memory reads.
- `PROFILER_ENABLE_LOAD_STORE_SKIPPING` (default: `1`): skip redundant load/store checks to reduce profiling overhead.
- `PROFILER_ENABLE_BLOCK_SAMPLING` (default: `1`): sample a subset of blocks to reduce profiling overhead.
- `PROFILER_DISABLE_BUFFER_LOAD_CHECK` (default: `0`): disable buffer load checks in the profiler.

## More Puzzles

If you're interested in fun puzzles to work with in Triton, do check out: [Triton Puzzles](https://github.com/srush/Triton-Puzzles)

## License

TileLens is licensed under the MIT License. See the [LICENSE](LICENSE) for details.

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

<a name="readme-top"></a>
# Triton-Viz: A Visualization Toolkit for programming with Triton
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="docs/logo.png" alt="Logo" width="320" height="320">
</div>
<br/>

Welcome to Triton-Viz, a visualization and profiling toolkit designed for deep learning applications. Built with the intention of making kernel programming in tile-based DSLs like Triton more intuitive.

Visit our [site](https://deep-learning-profiling-tools.github.io/triton-viz/) to see our tool in action!

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-of-triton-viz">Installation of Triton-Viz</a></li>
        <li><a href="#reproducing-the-inf2-nki-cost-model">Reproducing the Inf2 NKI cost model</a></li>
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

Triton-Viz helps developers inspect Triton kernels with visualization, profiling, and memory-safety analysis tools. It can run many examples through Triton's interpreter, so GPU access is not required for basic debugging workflows.


## Getting Started

### Prerequisites
- Python >= 3.10


### Installation of Triton-Viz

> **Windows Note:** Triton-viz depends on Triton, which can only be installed on Windows Subsystem for Linux (WSL). Once installed, follow below instructions in WSL.

Most users can install directly from PyPI:

```sh
pip install triton-viz
```

If you want to run examples from this repo, contribute, or build the web UI, install from source instead:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
uv sync # or "uv sync --extra test" if you're running tests
```

### Web UI Build

The PyPI package ships with prebuilt web UI assets in `triton_viz/static`, so
you do not need npm to run the visualizer. If you want to modify the web UI,
rebuild the TS sources:

```sh
npm install
npm run build:frontend
```

### Optional: Enable NKI Support

For PyPI installs, install with the `nki` extra and AWS Neuron repository:

```sh
pip install triton-viz[nki] --extra-index-url https://pip.repos.neuron.amazonaws.com
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

### Reproducing the Inf2 NKI cost model

The cost-model pipeline runs on an AWS Inferentia2 (Inf2) host
with an idle NeuronCore. The reported results were collected with the AWS
Neuron PyTorch 2.9 virtual environment at
`/opt/aws_neuronx_venv_pytorch_2_9`, `neuronx-cc 2.26.6360.0+6f180f47`, and a
separate Tilebench checkout. Compiler behavior is version-sensitive, so record
the output of `python -c "import neuronxcc; print(neuronxcc.__version__)"` when
reproducing or comparing results. The `nki` extra in this repository supplies
the Python-side NKI dependencies, but exact paper-number reproduction requires
the compiler version above.

From the repository root, activate the Neuron environment, put the repository
on `PYTHONPATH`, and run the three top-level stages:

```sh
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PYTHONPATH="$PWD"

python -m triton_viz.tools.nki_cost_model_pipeline collect --root /tmp/nki_cost_model_run \
  --tilebench-dir /path/to/Tilebench/benchmarks/operators
python -m triton_viz.tools.nki_cost_model_pipeline fit --root /tmp/nki_cost_model_run
python -m triton_viz.tools.nki_cost_model_pipeline evaluate --root /tmp/nki_cost_model_run
```

`collect` is the long-running hardware stage. It writes independent
microbenchmark/region controls under `microbench/` and `controls/`, and writes
Tilebench measurements under `holdouts/`. Both `collect` and `fit` use
`--skip-existing` / `--resume` so interrupted long runs can be continued.
`fit` reads only the control trees; it must not read `holdouts/`. `evaluate`
loads the frozen calibration and only then replays the holdouts. This
directory separation is an intentional train/holdout boundary, not merely an
output convention. Use `--dry-run` on any stage to inspect its commands
without executing them. `evaluate` asserts the formal FP32 holdout produces
exactly 35 rows and each FP32/BF16 full split produces exactly 120 rows before
writing `report.json`.

The region-control kinds `collect` requests are the 19 listed in
`nki_cost_model_pipeline.py` (`control_kinds`, around line 179), swept over
partitions {1,16,128} and free dims {128,512,1024,2048} in both dtypes, which
yields 156 `controls_p1` points. Seven of them --
`elementwise_mixed{,_masked}`, `reduce_broadcast`, `rsqrt_newton`,
`two_reductions`, `two_reductions_rsqrt`, `two_reductions_rsqrt_masked` -- were
absent from this list for a period even though the frozen experiment root
contains their data. Dropping them does not remove any Level-A rule, but it
leaves `reduction.two_or_more` with 5 calibration points instead of 16, and
layernorm's holdout error rises from 2.57% to 4.31%. They are required.

The main artifacts under the selected root are:

- `calibration/microbench.csv`, `dma_read_bf16_surface.csv`, `dma_write_fp32.csv`,
  `dma_write_bf16.csv`, `dma_read_large_free.csv`, `dma_transpose_surface.csv`:
  directional DMA bandwidth surfaces keyed by
  `(partition_count, free_bytes_per_partition)`.
- `calibration/dma_elapsed_frozen_v1.csv`: one global DMA descriptor-issue
  interval. A contiguous run coalesces into one bulk descriptor per partition
  and those are striped across the DMA engines, so their issue cost is
  negligible; a fragmented free axis costs one descriptor per element on the
  shared issue path. The resulting `descriptors * ns_per_descriptor` is a floor
  on DMA engine busy time and never lowers a bandwidth estimate.
- `calibration/compute.csv` (Level-B per-instruction VectorE/ScalarE cost),
  with optional `issue_floor_ns` for experimental calibration files: one shared
  engine/dtype initiation floor, `max(floor, startup + F * slope)`. Conflicting
  per-stream floors are rejected. Missing/blank columns preserve the affine
  path. This sets issue cost, not consumer latency (modelled separately).
  No shipped calibration enables the floor; operator-family validation is
  pending and this feature is not a promoted cost model.

- `calibration/structured_compute.csv` (Level-A expansion keyed by reusable structural
  grammar), `static_dma.csv`, and `strided_dma.csv`.
- `calibration/onchip_transfer_frozen_v1.csv`: measured PSUM/SBUF on-chip copy
  latency `startup + ns_per_free_elem * free`, keyed by `(engine, dtype)` with
  an explicit measured free-width domain. It replaces a hardcoded placeholder.
- `calibration/tensor_matmul_tiled.csv`: dtype-keyed TensorE
  `startup + flops / throughput` fallback.
- `calibration/tensor_source_geometry_frozen_v5.csv`: the TensorE surface
  actually used in production, a nonnegative control-only model over
  source-visible Dot count plus unique LHS, RHS and output tile counts. Equal
  FLOP counts with different tiled-Dot geometries are priced differently, which
  the FLOPs-only line could not do.
- `calibration/attention_pipeline_frozen_v1.csv`: TensorE busy for
  QK-normalize-PV Dot pipelines, interpolated in the value width. This is an
  engine occupancy surface, not a latency floor.
- `calibration/global_completion_frozen_v1.csv`: the single global NC
  completion model,
  `max(engine_busy) + beta * (sum(engine_busy) - max(engine_busy)) + offset`,
  with one `(beta, offset)` pair applied identically to every program. There is
  deliberately no structural, operator or grammar key.
- `evaluation/*.csv`: per-case predictions, hardware measurements, and the
  compute-only, compute-plus-DMA, scheduler-overlap and makespan-only
  ablations.
- `evaluation/report.json`: per-split and per-operator MAPE, the worst retained
  case, and a DMA-surface hit audit (`exact`/`interpolated`/`ood_clamped`).

There is exactly **one** completion term in the model. Earlier revisions carried
five per-structure completion floors (structured-grammar, norm, attention,
whole-program and strided) that looked up a same-class control program's
end-to-end time and used it as a lower bound, plus a second global
runtime-overhead floor. All six are gone: the per-structure tables because they
were lookups rather than models, and the runtime-overhead floor because it
estimated the same physical quantity as the completion offset and combining two
estimators with `max()` biased the prediction upward (an A/B over the 288
whole-program controls moved their MAPE from 10.913% to 7.959% and the median
signed error from +2.26% to -1.40%).

Engine occupancy for the whole-program grammar family is read from the control
surface by **linear interpolation** in the program's total source transfer
bytes, not by nearest neighbour: a partition-parallel engine's active time
grows linearly with free width, so snapping to the nearest control point errs
systematically in between. Values outside the measured range are clamped and
reported as `whole_program_routing_clamped`.

The DMA cost path is surface-based: the scheduler prices transfers from the
measured `(partition_count, free_bytes)` bandwidth surfaces, and the descriptor
issue floor above applies to the whole program's DMA resource occupancy. Out-of
-domain accesses are flagged (`ood_clamped`) rather than silently claimed as
in-domain. Compiler load CSE (`eliminate_redundant_hbm_loads`) and strided DMA
geometry are modeled independently of the surface and remain enabled.

Every calibration is fitted only on independent controls, gated by a
leave-one-out cross-validation before it may enter production, and refuses
target artifacts mechanically (`--artifact-role control`):

| calibration | CV protocol | result | gate |
|---|---|---|---|
| global completion | leave-one-free-dimension-out, 288 controls | 9.611% NC-p50 MAPE | <20% |
| DMA descriptor issue | leave-one-free-dimension-out, 30 controls | 14.498% | <20% |
| on-chip transfer | repeat-differenced, leave-one-width-out | 11.322% WAPE | <20% |
| whole-program routing | leave-one-free-dimension-out | Vector 3.85% / Scalar 3.70% WAPE | <20% |
| TensorE source geometry | leave-one-entire-suite-out, 6 suites | 6.34% BF16 / 1.45% FP32 WAPE | <20% |
| attention TensorE | leave-one-entire-width-grid-out | 0.075% WAPE | <20% |

The formal FP32 holdout contains 35 points at `rows=128` across eight operators:
`interleave`, `kl_divergence`, `layernorm`, `mul2`, `relu`, `rmsnorm`,
`sigmoid`, and `softmax`. All eight use free dimensions
`F={128,512,1024,2048}`; `interleave`, `layernorm`, and `rmsnorm` additionally
use `F=4096`, giving 35 points in total. No high-error point is removed, no
Level-B single-instruction constant is tuned on these points, and fitting does
not consume their measurements. The evaluated matrices are 120 FP32 and 120
BF16 points across `rows={1,16,128}` and `F={128,512,1024,2048,4096}`; BF16
strict evaluation rejects missing or cross-dtype compute fallback.

The same three-stage pipeline also evaluates two Tensor Engine splits and one
structural attention split in the identical collect/fit/evaluate flow (no
separate directory or standalone MAPE path):

- `tensor_fp32_v1` and `tensor_bf16_v1` add the Tilebench
  `matmul_fp32_fp16_fp8` kernel (5 points each). TensorE busy is accurate
  (whole-model TensorE WAPE 1.81%), but end-to-end NC-p50 MAPE is 19.31% (FP32)
  and 27.92% (BF16). The residual is the global overlap fraction, not engine
  occupancy: a software-pipelined tiled-Dot stream overlaps almost perfectly
  while the fitted global fraction is 0.55, and substituting the *measured*
  Vector and Scalar busy makes matmul worse, not better. A pipeline-depth
  overlap model that fixes this passes leave-one-entire-suite-out CV on a
  355-program control set spanning both regimes (10.22% versus the incumbent's
  13.99%) but degrades the norm families, so it is recorded as a rejected
  candidate rather than promoted.
- `attention_fp32_v1` uses the repository's real tiled attention kernel,
  `examples/nki_beta2/tiled_attention.py` (batch/head/M/KV tile loops, online
  softmax rescaling, and the full `dma_copy`/`nc_transpose`/`tensor_copy`/
  `nc_matmul` dataflow), traced by the `nki_beta2` frontend and benchmarked on
  Inf2 through the standalone NKI framework. Its 4-point NC-p50 MAPE is 27.18%.
  The residual is a DMA bandwidth-regime gap: its four HBM transfers hit the
  surface exactly yet achieve 24-56 GB/s where the surface reports 110-180, and
  the purpose-built isomorphic controls
  (`tensor_attention_pipeline_disjoint_{a,b}_v1`) achieve 116-180 GB/s, i.e.
  they do not reproduce the slowdown. It is reported rather than modelled,
  because closing it would require a target-specific correction. BF16 attention
  is unsupported because the example's second matmul mixes FP32 probabilities
  with BF16 V, which the beta2 interpreter rejects (`trace_unsupported`); Inf2
  PSUM capacity additionally constrains the compilable sweep to
  `m_size=n_size=128` with `dv_size={64,128,256,512}`. `report.json` carries
  these limits under `attention_coverage`.

With the compiler and environment above, the unified 254-point result is
**9.803% NC-p50 MAPE** (129 FP32 points at 9.239%, 125 BF16 at 10.386%), with
per-engine WAPE of 1.81% (TensorE), 7.73% (Vector) and 15.87% (Scalar). Thirteen
of the nineteen operator/dtype groups are under 15%; the six that are not
(matmul, attention, interleave, softmax BF16) have documented mechanism
diagnoses rather than tuned corrections.

The authoritative all-points headline (unique cases only, no split
double-counting) comes from the unified aggregator over the frozen replays:

```sh
python -m triton_viz.tools.nki_aggregate_unified_replay \
  /tmp/nki_cost_model_run/evaluation \
  --output-prefix /tmp/nki_cost_model_run/evaluation/unified
```

It writes `unified_cases.csv` and `unified_report.json` with the NC-p50 MAPE
and per-engine WAPE; `evaluation/report.json` remains the per-split,
per-operator breakdown.

See `microbench/inf2_nki/README.md` for individual microbenchmark commands,
schema details, and lower-level troubleshooting.

### Testing
* To run core Triton-viz tests, run `pytest tests/`.
* (if NKI installed) To run NKI-specific tests, run `pytest tests/ -m nki`.
* To run all tests (Triton + NKI), run `pytest tests/ -m ""`.
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
import triton_viz


@triton_viz.trace("sanitizer")  # also supports "tracer" and "profiler"
@triton.jit
def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, values)
```

Use the CLI wrappers to run an existing Python script without editing it. These
wrappers patch plain `@triton.jit` kernels, so use them with scripts that do not
already apply `@triton_viz.trace(...)`.

```sh
triton-sanitizer examples/sanitizer/oob_cli.py
triton-profiler examples/profiler/load_store_cli.py
triton-visualizer trace.tvz
```

For visualizer workflows, save a trace and launch the UI from Python:

```py
import triton_viz

triton_viz.save("trace.tvz")
triton_viz.launch()
```

## DSL Frontends

Triton is the default DSL frontend. NKI support is optional and selected with
the `frontend` argument:

```py
triton_viz.trace("tracer")  # Triton
triton_viz.trace("tracer", frontend="nki")  # NKI
triton_viz.trace("tracer", frontend="nki_beta2")  # NKI Beta 2
```

The runtime integration code lives under `triton_viz/core/frontend/`. NKI
simulation runtimes live under `triton_viz/core/simulation/`.

## Analysis Clients

Analyze kernels across visualization, profiling, and sanitization with a single line of code.

- Visualizer: currently supports load, store, and matmul operations for 1/2/3D tensors (more operations and dimensions coming soon).
- Profiler: flags non-unrolled loops, inefficient mask usage, and missing buffer_load optimizations while tracking load/store byte counts with low-overhead sampling.
- Sanitizer: symbolically checks tensor memory accesses for out-of-bounds errors and emits reports with tensor metadata, call stack, and expression trees; optional fake-memory storage avoids real reads.

### Save and load traces

```py
import triton_viz

triton_viz.save("trace.tvz")
triton_viz.load(
    "trace.tvz"
)  # automatically clears out existing records, use kwarg "append=True" to prevent this
triton_viz.launch()
```

CLI: `triton-visualizer trace.tvz`. The archive is a zip file containing `manifest.json` plus `tensors.npz`, and `triton_viz.load(...)` restores the normal trace state for existing consumers.

### Environment variables

Triton-Viz uses a small set of environment variables to configure runtime behavior. Unless noted, boolean flags are enabled only when set to `1`.

- `TRITON_VIZ_VERBOSE` (default: `0`): enable verbose logging and extra debug output.
- `TRITON_VIZ_NUM_SMS` (default: `1`): number of concurrent SMs to emulate for the CPU interpreter (min 1).
- `TRITON_VIZ_PORT` (default: `8000` with `share=True`, `5001` with `share=False`): port for the Flask server.
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

Triton-Viz is licensed under the MIT License. See the [LICENSE](LICENSE) for details.

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

When schedule-derived completion is enabled, `components_ns` also exposes
`schedule_payload_ns`, `schedule_residual_ns`, `schedule_static_floor_ns`, and
`schedule_global_floor_ns`. Final latency is the maximum of payload plus residual
and the two floors. Controls-only residual fitting can therefore retain the
static-DMA envelope instead of adding an offset to a floor that already overlaps
launch. These fields are absent from default reports.

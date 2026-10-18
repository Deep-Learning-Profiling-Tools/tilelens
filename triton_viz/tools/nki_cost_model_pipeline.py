"""Three-stage Inf2 NKI cost-model paper pipeline.

The stages deliberately keep calibration controls and Tilebench holdouts in
separate directories: ``collect`` gathers both, ``fit`` reads controls only,
and ``evaluate`` replays the frozen model on holdouts and reports ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from triton_viz.tools.nki_provenance import (
    collect_compiler_fingerprint,
    validate_model_manifest,
    write_model_manifest,
)

FORMAL_SPLITS = Path("microbench/inf2_nki/configs/formal_holdouts.json")
TOOL_MODULES = {
    "microbench.inf2_nki.harness.run_microbench": "microbench.inf2_nki.harness.run_microbench",
    "microbench.inf2_nki.profile_parser.export_csv": "microbench.inf2_nki.profile_parser.export_csv",
    "microbench.inf2_nki.profile_parser.fit_compute_calibration": "microbench.inf2_nki.profile_parser.fit_compute_calibration",
    "triton_viz.tools.nki_region_control_experiments": "triton_viz.tools.nki_region_control_experiments",
    "triton_viz.tools.nki_operator_experiments": "triton_viz.tools.nki_operator_experiments",
    "triton_viz.tools.nki_fit_structured_controls": "triton_viz.tools.nki_fit_structured_controls",
    "triton_viz.tools.nki_fit_structural_static_dma": "triton_viz.tools.nki_fit_structural_static_dma",
    "triton_viz.tools.nki_fit_strided_dma": "triton_viz.tools.nki_fit_strided_dma",
    "triton_viz.tools.nki_fit_tensor_source_geometry": "triton_viz.tools.nki_fit_tensor_source_geometry",
    "triton_viz.tools.nki_fit_attention_pipeline": "triton_viz.tools.nki_fit_attention_pipeline",
    "triton_viz.tools.nki_fit_global_completion": "triton_viz.tools.nki_fit_global_completion",
    "triton_viz.tools.nki_fit_dma_elapsed": "triton_viz.tools.nki_fit_dma_elapsed",
    "triton_viz.tools.nki_fit_onchip_copy": "triton_viz.tools.nki_fit_onchip_copy",
    "triton_viz.tools.nki_replay_operator_predictions": "triton_viz.tools.nki_replay_operator_predictions",
}


def _run(args: list[str], dry_run: bool) -> None:
    print("+", " ".join(args), flush=True)
    _validate_command_contract(args)
    if not dry_run:
        subprocess.run(args, check=True)


def _module(name: str, *args: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(arg) for arg in args)]


def _validate_command_contract(command: list[str]) -> None:
    """Parse a generated child command without executing its workload."""
    if len(command) < 4 or command[1] != "-m":
        raise ValueError(f"Not a Python module command: {command}")
    module_name = command[2]
    if module_name not in TOOL_MODULES:
        raise ValueError(f"Unknown pipeline child module: {module_name}")
    import importlib

    module = importlib.import_module(module_name)
    if hasattr(module, "build_parser"):
        module.build_parser().parse_args(command[3:])
        return

    # Capture the parser at the exact parse_args boundary. This validates the
    # real generated argv, then aborts before any hardware/filesystem work.
    from unittest.mock import patch

    original_parse_args = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse_and_stop(parser, args=None, namespace=None):
        original_parse_args(parser, command[3:], namespace)
        raise Parsed

    try:
        with patch.object(argparse.ArgumentParser, "parse_args", parse_and_stop):
            module.main(command[3:])
    except Parsed:
        return
    raise RuntimeError(f"{module_name}.main did not parse command-line arguments")


def _load_splits(path: Path = FORMAL_SPLITS) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "triton-viz.nki-formal-holdouts-v1":
        raise ValueError(f"Unsupported holdout split schema in {path}")
    return data["splits"]


def _split_case_count(split: dict) -> int:
    rows = split["rows"] if isinstance(split["rows"], list) else [split["rows"]]
    return len(rows) * sum(len(dims) for dims in split["operators"].values())


def _collect_split_commands(root: Path, tilebench: Path, name: str, split: dict):
    """Group operators by identical dimension lists without changing the split."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for op, dims in split["operators"].items():
        groups.setdefault(tuple(int(dim) for dim in dims), []).append(op)
    commands = []
    rows_values = split["rows"] if isinstance(split["rows"], list) else [split["rows"]]
    command_count = len(groups) * len(rows_values)
    index = 0
    for rows in rows_values:
        for dims, ops in sorted(groups.items()):
            output_name = name if command_count == 1 else f"{name}_{index}"
            commands.append(_module(
                "triton_viz.tools.nki_operator_experiments",
                "--output-dir",
                root / "holdouts" / output_name,
                "--tilebench-ops-dir",
                tilebench,
                "--ops",
                *ops,
                "--rows",
                int(rows),
                "--cols",
                *dims,
                "--dtype",
                split["dtype"],
                "--warmup",
                10,
                "--iters",
                100,
                # Targets are measured the same way the controls are: the median
                # of three compilations. Compilation is bimodal, so scoring a
                # median-calibrated model against a single compilation compares
                # the typical case to a lottery draw.
                "--hardware-trials",
                3,
                "--resume",
            ))
            index += 1
    return commands


def collect(root: Path, tilebench: Path, dry_run: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    configs = [
        "engine_lowering_sweep.json",
        "dma_partition_surface.json",
        "dma_partition_large_free.json",
        "dma_read_bf16_surface.json",
        "dma_transpose_surface.json",
        "dma_directional_dtype_canary.json",
        "dma_write_partition_surface.json",
        "dma_write_bf16_steady.json",
        "dma_strided_store_surface.json",
        "tensor_matmul_tiled_surface.json",
        "tensor_geometry_disjoint_v1.json",
        "tensor_geometry_disjoint_v3.json",
        "tensor_geometry_disjoint_v4.json",
        "tensor_geometry_disjoint_v5.json",
        "tensor_dot_count_low_disjoint_v1.json",
        "tensor_dot_count_low_disjoint_v2.json",
        "tensor_dot_count_low_disjoint_v3.json",
        "tensor_attention_pipeline_disjoint_a_v1.json",
        "tensor_attention_pipeline_disjoint_b_v1.json",
        "onchip_copy_disjoint_v2.json",
    ]
    for config in configs:
        run_id = Path(config).stem
        _run(
            _module(
                "microbench.inf2_nki.harness.run_microbench",
                "--config",
                Path("microbench/inf2_nki/configs") / config,
                "--output-root",
                root / "microbench",
                "--run-id",
                run_id,
                "--profile-export",
                "summary-json",
                "--skip-existing",
            ),
            dry_run,
        )

    # Region-control kinds.  The last seven were collected for the frozen root
    # by an invocation that predates this stage and were never requested here,
    # which left three Level-A rules badly under-determined: a collect without
    # them yields only 5 points for ``reduction.two_or_more``, 8 for
    # ``reduction.broadcast`` and 20 for ``elementwise.mixed`` (against 16, 16
    # and 32 in the frozen root).  Those are exactly the shapes layernorm and
    # rmsnorm lower into, and layernorm regresses 2.57% -> 4.31% without them.
    # Every rule still *appears* without these kinds, so this is a conditioning
    # problem rather than a coverage hole -- but it is a real one.
    control_kinds = [
        "elementwise_maximum",
        "elementwise_maximum_masked",
        "elementwise_multiply",
        "elementwise_multiply_masked",
        "elementwise_sigmoid",
        "elementwise_sigmoid_masked",
        "broadcast_multiply2",
        "broadcast_affine",
        "masked_log_reduction",
        "two_pass_reduce_affine",
        "two_pass_reduce_multiply",
        "softmax_reduction",
        "elementwise_mixed",
        "elementwise_mixed_masked",
        "reduce_broadcast",
        "rsqrt_newton",
        "two_reductions",
        "two_reductions_rsqrt",
        "two_reductions_rsqrt_masked",
    ]
    for partition_count in (1, 16, 128):
        _run(
            _module(
                "triton_viz.tools.nki_region_control_experiments",
                "--output-dir",
                root / f"controls_p{partition_count}",
                "--kinds",
                *control_kinds,
                "--free-dims",
                128,
                512,
                1024,
                2048,
                4096,
                "--dtypes",
                "float32",
                "bfloat16",
                "--p",
                partition_count,
                "--warmup",
                10,
                "--iters",
                100,
                # Control-only compiler audit: exports the Explorer parquet and
                # per-instruction source mapping that nki_fit_structured_controls
                # needs to accept a control row.  Runs after the measurement, so
                # it cannot perturb the timings, and it is never enabled for the
                # strict source-only stage2 controls or for any target artifact.
                "--postcompile-audit",
                "--resume",
            ),
            dry_run,
        )
    # Whole-program controls for the global NC completion model and the
    # whole-program engine-occupancy surface.  These run the same operator
    # programs as the holdouts but on a *disjoint* free-dimension grid
    # (192..3584 vs the holdouts' 128..4096), so no holdout column is ever
    # seen during calibration.
    whole_program_ops = [
        "interleave",
        "kl_divergence",
        "layernorm",
        "mul2",
        "relu",
        "rmsnorm",
        "sigmoid",
        "softmax",
    ]
    for name, dtype in (
        ("source_sequence_disjoint_fp32_v1", "float32"),
        ("source_sequence_disjoint_bf16_v1", "bfloat16"),
    ):
        _run(
            _module(
                "triton_viz.tools.nki_operator_experiments",
                "--output-dir",
                root / "stage2_controls" / name,
                "--tilebench-ops-dir",
                tilebench,
                "--ops",
                *whole_program_ops,
                "--rows",
                1,
                16,
                128,
                "--cols",
                192,
                384,
                768,
                1536,
                3072,
                3584,
                "--dtype",
                dtype,
                "--warmup",
                5,
                "--iters",
                20,
                # Same median-of-three protocol as every other measurement:
                # 16 of these 288 controls were caught in the slow compilation
                # mode, all on TensorE in layernorm/rmsnorm programs.
                "--hardware-trials",
                3,
                "--resume",
            ),
            dry_run,
        )

    splits = _load_splits()
    for name, split in splits.items():
        for command in _collect_split_commands(root, tilebench, name, split):
            _run(command, dry_run)


def _microbench_source_manifests(root: Path) -> list[Path]:
    manifests = []
    for run_manifest in sorted((root / "microbench").glob("*/run_manifest.json")):
        run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
        if run_data.get("num_ok") == run_data.get("num_benchmarks"):
            manifests.append(run_manifest)
            continue

        # A failed --resume attempt may legitimately use a different local
        # compiler environment and replace the suite-level manifest.  It must
        # not invalidate (or provide provenance for) successful, frozen
        # artifacts from an earlier run.  Case manifests prove which artifacts
        # succeeded but do not carry a compiler fingerprint, so exclude this
        # failed attempt from the model's collection-source set.
        case_manifests = sorted(run_manifest.parent.glob("**/manifest.json"))
        successful = [
            path
            for path in case_manifests
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
        ]
        if not successful:
            raise ValueError(
                f"Incomplete microbench run has no successful artifacts: {run_manifest}"
            )
    return manifests


def fit(root: Path, dry_run: bool, *, bf16_revisit_term: bool = False) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    canonical = calibration / "microbench.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench",
            "--output",
            canonical,
        ),
        dry_run,
    )
    dma_exports = {
        "dma_directional.csv": "dma_directional_dtype_canary",
        "dma_read_surface.csv": "dma_partition_surface",
        "dma_read_bf16_surface.csv": "dma_read_bf16_surface",
        "dma_transpose_surface.csv": "dma_transpose_surface",
        "dma_write_fp32.csv": "dma_write_partition_surface",
        "dma_write_bf16.csv": "dma_write_bf16_steady",
    }
    for output_name, run_name in dma_exports.items():
        _run(
            _module(
                "microbench.inf2_nki.profile_parser.export_csv",
                root / "microbench" / run_name,
                "--output",
                calibration / output_name,
            ),
            dry_run,
        )
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench" / "dma_partition_large_free",
            "--output",
            calibration / "dma_read_large_free.csv",
        ),
        dry_run,
    )
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench" / "tensor_matmul_tiled_surface",
            "--output",
            calibration / "tensor_matmul_tiled.csv",
        ),
        dry_run,
    )
    tensor_geometry_csv = calibration / "tensor_geometry_disjoint_v1.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench" / "tensor_geometry_disjoint_v1",
            "--output",
            tensor_geometry_csv,
        ),
        dry_run,
    )
    tensor_geometry_fit_csvs = []
    for run_name in (
        "tensor_geometry_disjoint_v3",
        "tensor_geometry_disjoint_v4",
        "tensor_geometry_disjoint_v5",
        "tensor_dot_count_low_disjoint_v1",
        "tensor_dot_count_low_disjoint_v2",
        "tensor_dot_count_low_disjoint_v3",
    ):
        output = calibration / f"{run_name}.csv"
        _run(
            _module(
                "microbench.inf2_nki.profile_parser.export_csv",
                root / "microbench" / run_name,
                "--output",
                output,
            ),
            dry_run,
        )
        tensor_geometry_fit_csvs.append(output)
    tensor_source_geometry = calibration / "tensor_source_geometry_frozen_v5.csv"
    _run(
        _module(
            "triton_viz.tools.nki_fit_tensor_source_geometry",
            *tensor_geometry_fit_csvs,
            *(["--bf16-revisit-term"] if bf16_revisit_term else []),
            "--artifact-role",
            "control",
            "--max-mean-wape",
            20,
            "--cv-output",
            calibration / "tensor_source_geometry_v5_strict_cv.json",
            "--output",
            tensor_source_geometry,
        ),
        dry_run,
    )
    attention_control_csvs = []
    for run_name in (
        "tensor_attention_pipeline_disjoint_a_v1",
        "tensor_attention_pipeline_disjoint_b_v1",
    ):
        output = calibration / f"{run_name}.csv"
        _run(
            _module(
                "microbench.inf2_nki.profile_parser.export_csv",
                root / "microbench" / run_name,
                "--output",
                output,
            ),
            dry_run,
        )
        attention_control_csvs.append(output)
    attention_pipeline = calibration / "attention_pipeline_frozen_v1.csv"
    _run(
        _module(
            "triton_viz.tools.nki_fit_attention_pipeline",
            *attention_control_csvs,
            "--artifact-role",
            "control",
            "--max-tensor-wape",
            20,
            "--cv-output",
            calibration / "attention_pipeline_strict_cv_v1.json",
            "--output",
            attention_pipeline,
        ),
        dry_run,
    )
    onchip_controls = calibration / "onchip_copy_disjoint_v2.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench" / "onchip_copy_disjoint_v2",
            "--output",
            onchip_controls,
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_onchip_copy",
            onchip_controls,
            "--artifact-role",
            "control",
            "--max-mean-wape",
            20,
            "--cv-output",
            calibration / "onchip_transfer_strict_cv_v1.json",
            "--output",
            calibration / "onchip_transfer_frozen_v1.csv",
        ),
        dry_run,
    )
    global_completion = calibration / "global_completion_frozen_v1.csv"
    global_completion_cv = calibration / "global_completion_strict_cv_v1.json"
    _run(
        _module(
            "triton_viz.tools.nki_fit_global_completion",
            root / "stage2_controls" / "source_sequence_disjoint_fp32_v1",
            root / "stage2_controls" / "source_sequence_disjoint_bf16_v1",
            "--artifact-role",
            "control",
            # The non-overlapped share of the residue grows with engine-load
            # imbalance; fitting that slope beats the single-constant form on
            # the same leave-one-free-dimension-out CV (8.16% vs 9.61%).
            "--fit-imbalance-slope",
            # Tree-structured partition setup/drain: launch cost grows with
            # log2(activated SBUF partitions), a spec-visible geometry term.
            "--fit-partition-offset",
            "--max-mean-mape",
            20,
            "--cv-output",
            global_completion_cv,
            "--output",
            global_completion,
        ),
        dry_run,
    )
    dma_elapsed = calibration / "dma_elapsed_frozen_v1.csv"
    _run(
        _module(
            "triton_viz.tools.nki_fit_dma_elapsed",
            root / "microbench" / "dma_strided_store_surface" / "results.jsonl",
            "--artifact-role",
            "control",
            "--global-completion-csv",
            global_completion,
            "--max-mean-mape",
            20,
            "--cv-output",
            calibration / "dma_elapsed_strict_cv_v1.json",
            "--output",
            dma_elapsed,
        ),
        dry_run,
    )
    compute = calibration / "compute.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.fit_compute_calibration",
            canonical,
            "--output",
            compute,
        ),
        dry_run,
    )
    structured = calibration / "structured_compute.csv"
    partitioned_control_roots = [
        root / "controls_p1",
        root / "controls_p16",
        root / "controls_p128",
    ]
    if dry_run or any(path.is_dir() for path in partitioned_control_roots):
        control_roots = partitioned_control_roots
        if not dry_run:
            missing = [path for path in control_roots if not path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    "Partition-aware controls are incomplete; missing "
                    + ", ".join(str(path) for path in missing)
                )
    else:
        control_roots = [root / "controls"]
    _run(
        _module(
            "triton_viz.tools.nki_fit_structured_controls",
            *control_roots,
            "--compute-calibration-csv",
            compute,
            "--artifact-role",
            "control",
            "--audit-output",
            calibration / "structured_compute_audit.csv",
            "--output",
            structured,
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_structural_static_dma",
            *control_roots,
            "--output",
            calibration / "static_dma.csv",
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_strided_dma",
            root / "microbench" / "dma_strided_store_surface" / "results.jsonl",
            "--output",
            calibration / "strided_dma.csv",
        ),
        dry_run,
    )
    if not dry_run:
        source_manifests = [
            *[
                control_root / "experiment_manifest.json"
                for control_root in control_roots
            ],
            *_microbench_source_manifests(root),
            # The whole-program controls behind the global NC completion model
            # and the whole-program engine-occupancy surface.  Earlier revisions
            # cited four tensor_attention_disjoint_* runs here instead; those
            # feed no production calibration and are not produced by collect,
            # so the bundle claimed provenance it did not have while omitting
            # the controls it actually depends on.
            *[
                root / "stage2_controls" / name / "experiment_manifest.json"
                for name in (
                    "source_sequence_disjoint_fp32_v1",
                    "source_sequence_disjoint_bf16_v1",
                )
            ],
        ]
        write_model_manifest(
            calibration,
            calibration_files=[
                canonical,
                calibration / "dma_directional.csv",
                calibration / "dma_read_surface.csv",
                calibration / "dma_read_bf16_surface.csv",
                calibration / "dma_read_large_free.csv",
                calibration / "dma_transpose_surface.csv",
                calibration / "dma_write_fp32.csv",
                calibration / "dma_write_bf16.csv",
                calibration / "tensor_matmul_tiled.csv",
                tensor_geometry_csv,
                *tensor_geometry_fit_csvs,
                tensor_source_geometry,
                calibration / "tensor_source_geometry_v5_strict_cv.json",
                *attention_control_csvs,
                attention_pipeline,
                calibration / "attention_pipeline_strict_cv_v1.json",
                global_completion,
                global_completion_cv,
                dma_elapsed,
                calibration / "dma_elapsed_strict_cv_v1.json",
                # The on-chip PSUM/SBUF copy surface is validated by evaluate,
                # so the bundle has to declare it.  It was added to production
                # and to the evaluate check but never here, which only stayed
                # invisible because evaluate was not run end-to-end afterwards.
                calibration / "onchip_transfer_frozen_v1.csv",
                calibration / "onchip_transfer_strict_cv_v1.json",
                compute,
                structured,
                calibration / "static_dma.csv",
                calibration / "strided_dma.csv",
            ],
            source_manifests=source_manifests,
            split_file=FORMAL_SPLITS,
            payload_definition="raw_engine_active_no_runtime_baseline",
        )


def _replay_args(root: Path, holdout: Path, output: Path, dtype: str) -> list[str]:
    calibration = root / "calibration"
    write_csv = (
        calibration / "dma_write_bf16.csv"
        if dtype == "bfloat16"
        else calibration / "dma_write_fp32.csv"
    )
    args = _module(
        "triton_viz.tools.nki_replay_operator_predictions",
        holdout,
        "--dma-read-surface-csv",
        (
            calibration / "dma_read_bf16_surface.csv"
            if dtype == "bfloat16"
            else calibration / "microbench.csv"
        ),
        "--dma-write-surface-csv",
        write_csv,
        "--compute-calibration-csv",
        calibration / "compute.csv",
        "--structured-control-csv",
        calibration / "structured_compute.csv",
        "--tensor-calibration-csv",
        calibration / "tensor_matmul_tiled.csv",
        "--tensor-source-geometry-csv",
        calibration / "tensor_source_geometry_frozen_v5.csv",
        "--attention-pipeline-calibration-csv",
        calibration / "attention_pipeline_frozen_v1.csv",
        "--global-completion-csv",
        calibration / "global_completion_frozen_v1.csv",
        "--dma-elapsed-csv",
        calibration / "dma_elapsed_frozen_v1.csv",
        "--onchip-transfer-csv",
        calibration / "onchip_transfer_frozen_v1.csv",
        "--structural-static-dma-csv",
        calibration / "static_dma.csv",
        "--output",
        output,
    )
    if dtype == "float32":
        args[args.index("--compute-calibration-csv"):args.index("--compute-calibration-csv")] = [
            "--dma-transpose-surface-csv",
            str(calibration / "dma_transpose_surface.csv"),
        ]
    return args


def _evaluate_replays(
    root: Path,
    replay_dir: Path,
    splits: dict,
    *,
    model_name: str,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    for split_name, split in splits.items():
        holdout_dirs = sorted((root / "holdouts").glob(f"{split_name}*"))
        if dry_run and not holdout_dirs:
            holdout_dirs = [
                root / "holdouts" / split_name
                if len(_collect_split_commands(root, Path("."), split_name, split)) == 1
                else root / "holdouts" / f"{split_name}_{index}"
                for index in range(
                    len(_collect_split_commands(root, Path("."), split_name, split))
                )
            ]
        for holdout in holdout_dirs:
            output = replay_dir / f"{model_name}_{holdout.name}.csv"
            args = _replay_args(root, holdout, output, split["dtype"])
            args[args.index("--output"):args.index("--output")] = [
                "--strict-calibration",
            ]
            if split_name in {"formal_fp32_v1", "full_fp32_v1", "full_bf16_v1"}:
                # These splits are the whole-program grammar family, so they
                # replay with the same strided-DMA busy surface and the same
                # control-routed engine occupancy the frozen result used.
                suffix = "bf16" if split["dtype"] == "bfloat16" else "fp32"
                args[args.index("--output"):args.index("--output")] = [
                    "--strided-dma-csv",
                    str(root / "calibration" / "strided_dma.csv"),
                    "--whole-program-control-root",
                    str(
                        root
                        / "stage2_controls"
                        / f"source_sequence_disjoint_{suffix}_v1"
                    ),
                ]
            if extra_args:
                args[args.index("--output"):args.index("--output")] = list(extra_args)
            _run(args, dry_run)
            outputs.append((split_name, output))
    return outputs


def evaluate(root: Path, dry_run: bool, instruction_latency_csv: Path | None = None) -> None:
    replay_dir = root / "evaluation"
    replay_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        calibration = root / "calibration"
        validate_model_manifest(
            calibration / "model_manifest.json",
            calibration_files=[
                calibration / "microbench.csv",
                calibration / "dma_directional.csv",
                calibration / "dma_read_surface.csv",
                calibration / "dma_read_bf16_surface.csv",
                calibration / "dma_read_large_free.csv",
                calibration / "dma_transpose_surface.csv",
                calibration / "dma_write_fp32.csv",
                calibration / "dma_write_bf16.csv",
                calibration / "tensor_matmul_tiled.csv",
                calibration / "tensor_geometry_disjoint_v3.csv",
                calibration / "tensor_geometry_disjoint_v4.csv",
                calibration / "tensor_geometry_disjoint_v5.csv",
                calibration / "tensor_dot_count_low_disjoint_v1.csv",
                calibration / "tensor_dot_count_low_disjoint_v2.csv",
                calibration / "tensor_dot_count_low_disjoint_v3.csv",
                calibration / "tensor_source_geometry_frozen_v5.csv",
                calibration / "tensor_source_geometry_v5_strict_cv.json",
                calibration / "tensor_attention_pipeline_disjoint_a_v1.csv",
                calibration / "tensor_attention_pipeline_disjoint_b_v1.csv",
                calibration / "attention_pipeline_frozen_v1.csv",
                calibration / "attention_pipeline_strict_cv_v1.json",
                calibration / "global_completion_frozen_v1.csv",
                calibration / "global_completion_strict_cv_v1.json",
                calibration / "dma_elapsed_frozen_v1.csv",
                calibration / "dma_elapsed_strict_cv_v1.json",
                calibration / "onchip_transfer_frozen_v1.csv",
                calibration / "onchip_transfer_strict_cv_v1.json",
                calibration / "compute.csv",
                calibration / "structured_compute.csv",
                calibration / "static_dma.csv",
                calibration / "strided_dma.csv",
            ],
            current_fingerprint=collect_compiler_fingerprint(
                Path(__file__).resolve().parents[2]
            ),
        )
    splits = _load_splits()
    outputs = _evaluate_replays(
        root,
        replay_dir,
        splits,
        model_name="surface",
        dry_run=dry_run,
        extra_args=(["--instruction-latency-csv", str(instruction_latency_csv)]
                    if instruction_latency_csv else None),
    )
    if dry_run:
        return

    rows_by_split = {
        split_name: [
            row
            for name, path in outputs
            if name == split_name
            for row in csv.DictReader(path.open())
        ]
        for split_name in splits
    }
    formal_rows = rows_by_split["formal_fp32_v1"]
    expected = _split_case_count(splits["formal_fp32_v1"])
    if len(formal_rows) != expected:
        raise ValueError(
            f"formal_fp32_v1 expected {expected} successful cases, got {len(formal_rows)}"
        )
    full_rows = rows_by_split["full_fp32_v1"]
    full_expected = _split_case_count(splits["full_fp32_v1"])
    if len(full_rows) != full_expected or full_expected != 120:
        raise ValueError(
            f"full_fp32_v1 expected exactly 120 successful cases, got {len(full_rows)}"
        )
    full_bf16_rows = rows_by_split["full_bf16_v1"]
    full_bf16_expected = _split_case_count(splits["full_bf16_v1"])
    if len(full_bf16_rows) != full_bf16_expected or full_bf16_expected != 120:
        raise ValueError(
            f"full_bf16_v1 expected exactly 120 successful cases, "
            f"got {len(full_bf16_rows)}"
        )
    tensor_rows = [
        row
        for split_name in ("tensor_fp32_v1", "tensor_bf16_v1")
        for row in rows_by_split.get(split_name, [])
    ]
    tensor_expected = sum(
        _split_case_count(splits[name])
        for name in ("tensor_fp32_v1", "tensor_bf16_v1")
        if name in splits
    )
    if len(tensor_rows) != tensor_expected:
        raise ValueError(
            f"TensorE holdouts expected {tensor_expected} successful cases, "
            f"got {len(tensor_rows)}"
        )
    attention_rows = rows_by_split.get("attention_fp32_v1", [])
    attention_expected = _split_case_count(splits["attention_fp32_v1"])
    if len(attention_rows) != attention_expected:
        raise ValueError(
            f"attention_fp32_v1 expected {attention_expected} successful cases, "
            f"got {len(attention_rows)}"
        )
    invalid_bf16_matches = sorted(
        {
            row["calibration_match"]
            for row in full_bf16_rows
            if "fallback" in row["calibration_match"]
            or "missing" in row["calibration_match"]
        }
    )
    if invalid_bf16_matches:
        raise ValueError(
            "BF16 strict evaluation used invalid compute calibration matches: "
            f"{invalid_bf16_matches}"
        )
    stages = {
        "compute_only_mape_pct": "compute_only_error_pct",
        "compute_plus_dma_mape_pct": "compute_dma_error_pct",
        "resource_overlap_mape_pct": "resource_overlap_error_pct",
        "makespan_only_mape_pct": "makespan_only_error_pct",
        "final_nc_p50_mape_pct": "nc_error_pct",
    }
    auxiliary_rows = rows_by_split.get("auxiliary_bf16_v1", [])
    report = {
        "formal_fp32_cases": len(formal_rows),
        "full_fp32_cases": len(full_rows),
        "full_bf16_cases": len(full_bf16_rows),
        "tensor_cases": len(tensor_rows),
        "attention_cases": len(attention_rows),
        "auxiliary_bf16_cases": len(auxiliary_rows),
    }
    report["full_fp32_nc_p50_mape_pct"] = statistics.mean(
        abs(float(row["nc_error_pct"])) for row in full_rows
    )
    report["full_fp32_rows_mape_pct"] = {
        str(rows): statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in full_rows
            if int(row["rows"]) == rows
        )
        for rows in (1, 16, 128)
    }
    report["full_fp32_dma_surface_event_counts"] = {
        match: sum(int(row[field]) for row in full_rows)
        for match, field in {
            "exact": "dma_surface_exact_count",
            "interpolated": "dma_surface_interpolated_count",
            "ood_clamped": "dma_surface_ood_count",
        }.items()
    }
    report["full_bf16_nc_p50_mape_pct"] = statistics.mean(
        abs(float(row["nc_error_pct"])) for row in full_bf16_rows
    )
    report["full_bf16_rows_mape_pct"] = {
        str(rows): statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in full_bf16_rows
            if int(row["rows"]) == rows
        )
        for rows in (1, 16, 128)
    }
    report["full_bf16_operator_mape_pct"] = {
        operator: statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in full_bf16_rows
            if row["op"] == operator
        )
        for operator in sorted({row["op"] for row in full_bf16_rows})
    }
    report["full_bf16_dma_surface_event_counts"] = {
        match: sum(int(row[field]) for row in full_bf16_rows)
        for match, field in {
            "exact": "dma_surface_exact_count",
            "interpolated": "dma_surface_interpolated_count",
            "ood_clamped": "dma_surface_ood_count",
        }.items()
    }
    report["full_bf16_compute_calibration_matches"] = sorted(
        {row["calibration_match"] for row in full_bf16_rows}
    )
    mechanism_rows = full_rows + full_bf16_rows + tensor_rows + attention_rows
    def mechanism_wape(predicted_field: str, actual_field: str) -> float | None:
        pairs = [
            (float(row[predicted_field]), float(row[actual_field]))
            for row in mechanism_rows
            if row.get(predicted_field) not in (None, "")
            and row.get(actual_field) not in (None, "")
        ]
        denominator = sum(actual for _, actual in pairs)
        if not pairs or denominator <= 0:
            return None
        return sum(abs(predicted - actual) for predicted, actual in pairs) / denominator * 100.0

    mechanism_fields = {
        "dynamic_dma": ("predicted_dynamic_dma_us", "hardware_dynamic_dma_us"),
        "static_dma": ("predicted_static_dma_us", "hardware_static_dma_us"),
        "vector_payload": (
            "predicted_vector_payload_us",
            "hardware_vector_payload_us",
        ),
        "scalar_payload": (
            "predicted_scalar_payload_us",
            "hardware_scalar_payload_us",
        ),
        "gpsimd_payload": (
            "predicted_gpsimd_payload_us",
            "hardware_gpsimd_payload_us",
        ),
        "tensor": ("predicted_tensor_us", "hardware_tensor_active_us"),
    }
    report["mechanism_busy_wape_pct"] = {
        name: value
        for name, fields in mechanism_fields.items()
        if (value := mechanism_wape(*fields)) is not None
    }
    report["mechanism_busy_metric"] = (
        "WAPE=sum(abs(predicted-actual))/sum(actual); zero/small actual rows retained"
    )
    report["mechanism_coverage"] = {}
    for name, field, coverage_field in (
        ("vector", "vector_payload_error_pct", "micro_dag_vector_covered"),
        ("scalar", "scalar_payload_error_pct", "micro_dag_scalar_covered"),
        ("gpsimd", "gpsimd_payload_error_pct", "micro_dag_gpsimd_covered"),
        ("tensor", "tensor_error_pct", "micro_dag_tensor_covered"),
        ("static_dma", "static_dma_error_pct", "micro_dag_static_dma_covered"),
    ):
        measured = [
            row
            for row in mechanism_rows
            if row.get(field) not in (None, "")
        ]
        covered = [
            row for row in measured if int(row.get(coverage_field) or 0)
        ]
        report["mechanism_coverage"][name] = {
            "measured_cases": len(measured),
            "covered_cases": len(covered),
            "coverage_rate": len(covered) / len(measured) if measured else 0.0,
            "covered_case_mape_pct": (
                statistics.mean(abs(float(row[field])) for row in covered)
                if covered
                else None
            ),
        }
    report["micro_dag_audit"] = {
        "unsupported_engine_events": sum(
            int(row.get("micro_dag_unsupported_engine_events") or 0)
            for row in mechanism_rows
        ),
        "timing_exact_events": sum(
            int(row.get("micro_dag_timing_exact_count") or 0)
            for row in mechanism_rows
        ),
        "timing_interpolated_events": sum(
            int(row.get("micro_dag_timing_interpolated_count") or 0)
            for row in mechanism_rows
        ),
        "timing_aggregate_events": sum(
            int(row.get("micro_dag_timing_aggregate_count") or 0)
            for row in mechanism_rows
        ),
    }
    report["tensor_source_geometry_ood_count"] = sum(
        int(row.get("tensor_source_geometry_ood_count") or 0)
        for row in tensor_rows
    )
    report["tensor_nc_p50_mape_pct"] = statistics.mean(
        abs(float(row["nc_error_pct"])) for row in tensor_rows
    )
    report["tensor_busy_mape_pct"] = statistics.mean(
        abs(float(row["tensor_error_pct"])) for row in tensor_rows
    )
    report["tensor_operator_mape_pct"] = {
        f"{operator}/{dtype}": statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in tensor_rows
            if row["op"] == operator and row["dtype"] == dtype
        )
        for operator, dtype in {
            (row["op"], row["dtype"]) for row in tensor_rows
        }
    }
    report["tensor_flops_domain_ood_count"] = sum(
        int(row["tensor_flops_domain_ood_count"]) for row in tensor_rows
    )
    report["attention_nc_p50_mape_pct"] = statistics.mean(
        abs(float(row["nc_error_pct"])) for row in attention_rows
    )
    report["attention_tensor_busy_mape_pct"] = statistics.mean(
        abs(float(row["tensor_error_pct"]))
        for row in attention_rows
        if row.get("tensor_error_pct") not in (None, "")
    )
    report["attention_dma_error_mape_pct"] = statistics.mean(
        abs(float(row["dma_error_pct"])) for row in attention_rows
    )
    report["attention_operator_mape_pct"] = {
        operator: statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in attention_rows
            if row["op"] == operator
        )
        for operator in sorted({row["op"] for row in attention_rows})
    }
    report["attention_tensor_flops_domain_ood_count"] = sum(
        int(row["tensor_flops_domain_ood_count"]) for row in attention_rows
    )
    report["attention_coverage"] = {
        "beta2_frontend": "nki_beta2",
        "unsupported_bfloat16": "mixed-precision PV matmul rejected by the beta2 interpreter",
        "nc_reference": "median of framework device-side latency samples",
    }
    if auxiliary_rows:
        report["auxiliary_bf16_nc_p50_mape_pct"] = statistics.mean(
            abs(float(row["nc_error_pct"])) for row in auxiliary_rows
        )
    for name, field in stages.items():
        report[name] = statistics.mean(
            abs(float(row[field])) for row in formal_rows
        )
    report["formal_fp32_dma_surface_event_counts"] = {
        "exact": sum(
            int(row["dma_surface_exact_count"]) for row in formal_rows
        ),
        "interpolated": sum(
            int(row["dma_surface_interpolated_count"]) for row in formal_rows
        ),
        "ood_clamped": sum(
            int(row["dma_surface_ood_count"]) for row in formal_rows
        ),
    }
    report["formal_fp32_dma_surface_max_log_distance"] = max(
        float(row["dma_surface_max_log_distance"]) for row in formal_rows
    )
    report["formal_fp32_global_completion_audit"] = {
        "activated_cases": sum(
            int(row["global_completion_activated"]) for row in formal_rows
        ),
        "activation_rate": statistics.mean(
            int(row["global_completion_activated"]) for row in formal_rows
        ),
    }
    report["formal_fp32_operator_mape_pct"] = {
        operator: statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in formal_rows
            if row["op"] == operator
        )
        for operator in sorted({row["op"] for row in formal_rows})
    }
    worst = max(formal_rows, key=lambda row: abs(float(row["nc_error_pct"])))
    report["worst_case"] = {
        "case": worst["case"],
        "error_pct": float(worst["nc_error_pct"]),
    }
    (replay_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "fit", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--tilebench-dir",
        type=Path,
        default=Path("/home/ubuntu/Tilebench/benchmarks/operators"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instruction-latency-csv", type=Path, default=None,
                        help="Evaluate only: opt into consumer-visible instruction latency (controls-only CSV).")
    parser.add_argument("--bf16-revisit-term", action="store_true",
                        help="Fit only: opt into the experimental BF16 Tensor geometry model.")
    args = parser.parse_args(argv)
    if args.bf16_revisit_term and args.stage != "fit":
        parser.error("--bf16-revisit-term is only valid for fit")
    if args.stage == "collect":
        collect(args.root.resolve(), args.tilebench_dir.resolve(), args.dry_run)
    elif args.stage == "fit":
        fit(args.root.resolve(), args.dry_run, bf16_revisit_term=args.bf16_revisit_term)
    else:
        evaluate(args.root.resolve(), args.dry_run,
                 args.instruction_latency_csv.resolve() if args.instruction_latency_csv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os


def _get_env(env: str, default: str) -> str:
    """Prefer TileLens settings, falling back to the former variable names."""
    if env.startswith("TILELENS_"):
        legacy = env.replace("TILELENS_", "TRITON_VIZ_", 1)
        return os.getenv(env, os.getenv(legacy, default))
    return os.getenv(env, default)


def _is_one(env: str, default: str = "0") -> bool:
    return _get_env(env, default) == "1"


def _get_int_env(env: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(_get_env(env, str(default)))
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


class Config:
    """
    Runtime configuration loaded from environment variables.

    Fields map to environment variables with string values where "1" enables a flag:
    - verbose: TILELENS_VERBOSE, enables verbose logging.
    - num_sms: TILELENS_NUM_SMS, emulated concurrent SMs in the Triton CPU
      interpreter (min 1).
    - enable_sanitizer: ENABLE_SANITIZER, toggles the sanitizer pipeline.
    - enable_race_detector: ENABLE_RACE_DETECTOR, toggles the data-race
      detector pipeline.
    - enable_timing: ENABLE_TIMING, collects timing info during execution.
    - virtual_memory: SANITIZER_ENABLE_FAKE_TENSOR, uses fake tensor storage in
      sanitizer runs to avoid real memory reads.
    - profiler_enable_load_store_skipping: PROFILER_ENABLE_LOAD_STORE_SKIPPING,
      replaces loads/stores/dot with no-ops returning zeros to speed profiling.
    - profiler_enable_block_sampling: PROFILER_ENABLE_BLOCK_SAMPLING, samples a
      subset of blocks to reduce profiling overhead.
    - profiler_disable_buffer_load_check: PROFILER_DISABLE_BUFFER_LOAD_CHECK,
      disables buffer load checks in the profiler.
    - symbolic_per_element_warn_threshold:
      SYMBOLIC_PER_ELEMENT_WARN_THRESHOLD, element count above which the
      shared SymbolicClient (sanitizer + race detector) emits a UserWarning
      before falling back to per-element address enumeration for
      non-unit-inner-stride tensors. Behavior is unchanged; the warning just
      signals that the symbolic solver may slow down. Set to 0 to disable
      the warning entirely.
    - sanitizer_report_max_segments: SANITIZER_REPORT_MAX_SEGMENTS, max
      number of address segments to list verbatim in the OOB report before
      truncating to a head/tail summary. Affects display only (min 2).
    """

    def __init__(self) -> None:
        self.cli_active: bool = False
        self.reset()

    def reset(self) -> None:
        """Reload configuration from environment variables and apply defaults."""
        self.verbose: bool = _is_one("TILELENS_VERBOSE")
        self.num_sms: int = _get_int_env("TILELENS_NUM_SMS", 1, minimum=1)
        self.enable_sanitizer: bool = _is_one("ENABLE_SANITIZER", "1")
        self.enable_race_detector: bool = _is_one("ENABLE_RACE_DETECTOR", "1")
        self.enable_timing: bool = _is_one("ENABLE_TIMING")
        self.virtual_memory: bool = _is_one("SANITIZER_ENABLE_FAKE_TENSOR")
        self.profiler_enable_load_store_skipping: bool = _is_one(
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING", "1"
        )
        self.profiler_enable_block_sampling: bool = _is_one(
            "PROFILER_ENABLE_BLOCK_SAMPLING", "1"
        )
        self.profiler_disable_buffer_load_check: bool = _is_one(
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK"
        )
        self.symbolic_per_element_warn_threshold: int = _get_int_env(
            "SYMBOLIC_PER_ELEMENT_WARN_THRESHOLD", 8192, minimum=0
        )
        self.sanitizer_report_max_segments: int = _get_int_env(
            "SANITIZER_REPORT_MAX_SEGMENTS", 8, minimum=2
        )


config = Config()

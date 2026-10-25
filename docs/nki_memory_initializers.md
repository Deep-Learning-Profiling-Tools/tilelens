# Beta2 memory initializers and allocation metadata

The beta2 interpreter supports `nisa.memset(dst, value, engine=...)` for on-chip
storage. It writes the destination numerically and emits a source initializer
record with no tensor reads. Vector and GpSimd requests are retained; an unspecified
engine remains unknown. Non-scalar values and off-chip destinations are rejected.

`records_to_events` exports these records as `memory_fill`, separately from
arithmetic `compute` events. They carry:

* Destination storage, shape, dtype, exact output runs and memory space.
* Declared root allocation identity, shape and partition axis, including when a
  fill targets a view of that allocation.
* Compile-time scalar value and requested engine.
* `timing_status="unpriced_initializer"`.

This exposes initialization writes for dependency/lifetime analysis. It does not
provide an independently calibrated fill timing law. The analytical cost model
rejects an unpriced initializer explicitly rather than giving it zero or an
unrelated arithmetic price. Initializer metadata is not inherited by subsequent
in-place arithmetic on the same array.

The interpreter exposes Gen2 `tile_size.psum_num_banks=8`,
`psum_bank_fmax=512` and `psum_bank_fmax_bytes=2048`. Other targets and configurations
that reserve a bank for PE transpose require separate target handling.

For a transfer that changes element size, trace events retain `src_bytes` and
`dst_bytes` and compute each endpoint's exact byte spans using its own width.
The existing `bytes` field continues to represent destination bytes. This matters
for FP32 PSUM-to-BF16 SBUF copies; applying destination element width to source
addresses incorrectly truncates the source footprint.

These are tracing features, not a complete physical allocator, compiler ordering
model or timing calibration. Programs without initializers keep their existing
trace/pricing behavior; a representative default rowdot report is unchanged
apart from elapsed-time metadata.

Focused tests are in `tests/nki/test_nki_beta2_memory_fill.py`:

```
python -m pytest -q -m nki tests/nki/test_nki_beta2_memory_fill.py
```

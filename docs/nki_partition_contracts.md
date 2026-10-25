# Executed partition-extent checks (opt-in)

The beta2 interpreter can check that each executed on-chip ISA operand has at most
128 partitions. This catches invalid attempts to improve reuse by placing an
entire 256/512/1024-row tensor on SBUF's partition axis.

```python
from triton_viz.core.simulation.nki_isa_contracts import partition_contract_checks

with partition_contract_checks():
    triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)[grid](*inputs)
```

`ISAContractError.finding` carries the operation, operand, buffer, concrete shape,
partition axis, extent, limit and generic layout guidance. Explicit `nl.par_dim`
markers take precedence; unmarked shapes use the ISA's leading-axis default.
HBM dimensions and large free dimensions are not rejected by this check. Nested
contexts and exceptions restore the previous policy. Defaults remain unchanged.

The check operates at operand use, not by summing declared allocations. It is
partial: it does not establish PSUM/SBUF capacity, compiler placement, legality
of unexecuted paths, other ISA contracts, or compilation success. A first earlier
interpreter error can prevent reaching a later partition violation. A successful
checked interpretation must not be presented as a full compiler legality proof.

## SDK-faithful basic views (extended opt-in implementation)

Inside the same context, bare SBUF/PSUM allocations keep their leading partition
axis metadata. Integer selection retains that axis with extent one, while HBM
integer indexing drops the selected dimension; on-chip views remain at least 2D
and HBM views at least 1D. Basic slices, ellipsis and negative indices preserve
storage sharing. A legal single-partition matmul remains executable.

For example `(Kblocks, 128, M)[k, :, :]` on SBUF is `(1, 128, M)`, not `(128, M)`.
The checker reports `matmul_partition_mismatch` when this size-one partition is
paired with a 128-partition operand, including both shapes/extents and guidance.
Batch/reuse dimensions should not silently become contraction partitions.

Executed direct DMA follows the installed SDK Python validation's equal-element
rule in this opt-in mode, so layout/rank differences do not mask later contraction
errors. The historical shape-equality restriction remains outside the context.
This does not validate every backend AP, allocation or placement constraint.

Tests compare 492 basic views against the installed SDK's actual `NkiTensor`
placeholder views, including shape, offset, values, storage aliasing and observable
strides. A separate SDK DMA validation test stubs backend materialization/NC2 context,
while leaving the SDK geometry validation intact. The parity claim covers ordinary
leading-axis partitions and HBM, not legacy nonleading `par_dim` compatibility.
Defaults and the earlier frozen constraint-only branch remain separate.

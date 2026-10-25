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

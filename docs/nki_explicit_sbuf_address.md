# Explicit SBUF addresses in the NKI interpreter

The beta2 interpreter accepts a nonnegative `(partition_offset, free_offset)`
`address` argument for SBUF allocations and executes the resulting code with
separate virtual NumPy storage. This matches the storage abstraction used by
the local NKI SDK simulator. It does not emulate physical aliasing, allocation
capacity, compiler placement, or instruction scheduling. Explicit PSUM/HBM
addresses and address combined with a `value` override are not supported.

Tracing marks the allocation even when it is never used, and also marks
operations that touch it. The cost model refuses to price a trace containing
such a marker. A caller must likewise treat numerical correctness, output
coverage and provenance under these virtual addresses as unverified for the
physical kernel, until it has an independent physical-allocation model or a
device check. Passing a virtual interpreter comparison is not device evidence.

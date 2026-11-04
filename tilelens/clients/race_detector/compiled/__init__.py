"""The compiled race detector: race analysis over host-compiled TTIR.

``leaves`` gives the TTIR reader's terms their meaning for the two-copy
solver (``RaceLeaves``), ``awaits`` and ``fences`` hold the race detector's
await and fence rules over the reader's neutral facts, ``global_records``
lowers an ``AccessGraph`` to the solver's records, and ``analysis`` picks
the proof rung for one compiled kernel.
"""

"""Generate small, observable controls for explicit-engine SBUF fills.

Every fill uses a distinct literal and is written to a distinct output slice;
hardware opcode counts still need auditing before any timing is fitted.  No
device measurements are performed by this module.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FillControl:
    partitions: int
    width: int
    engine: str
    split: str
    repeats: int = 8

    def __post_init__(self):
        if (type(self.partitions) is not int or not 1<=self.partitions<=128 or
            type(self.width) is not int or not 1<=self.width<=4096 or
            type(self.repeats) is not int or not 1<=self.repeats<=16 or
            self.engine not in ('vector','gpsimd') or self.split not in ('fit','validation')):
            raise ValueError('invalid fill control')

    @property
    def name(self):
        return f'{self.engine}_p{self.partitions}_f{self.width}_r{self.repeats}'

    def source(self):
        p,f=self.partitions,self.width
        lines=['import nki.isa as nisa','import nki.language as nl','','def kernel(out):',
               f'    tile=nl.ndarray(({p},{f}),dtype=nl.float32,buffer=nl.sbuf)']
        for i in range(self.repeats):
            lines.append(f'    nisa.memset(dst=tile,value={float(i+1)},engine=nisa.engine.{self.engine})')
            lines.append(f'    nisa.dma_copy(dst=out[{i*p}:{(i+1)*p},:],src=tile)')
        return '\n'.join(lines+['    return out',''])

    def task(self):
        p,f,n=self.partitions,self.width,self.repeats
        return f'''import numpy as np
frontend='nki_beta2'
grid=(1,)
output_name='out'
rtol=0.0
atol=0.0
def make_inputs(seed):
    return {{'out':np.zeros(({p*n},{f}),dtype=np.float32)}}
def reference(values):
    return np.repeat(np.arange(1,{n+1},dtype=np.float32),{p})[:,None] * np.ones((1,{f}),dtype=np.float32)
'''


def controls():
    """Fit/validation points are fixed before any device captures."""
    points=[(128,64,'fit'),(128,256,'fit'),(128,1024,'fit'),(128,4096,'fit'),
            (32,512,'fit'),(128,768,'validation'),(64,512,'validation')]
    return tuple(FillControl(p,f,engine,split) for engine in ('vector','gpsimd')
                 for p,f,split in points)

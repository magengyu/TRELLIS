"""Create synthetic boxes and dummy conditions for execution checks, not useful training."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resolution',type=int,default=8)
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--cond-channels',type=int,default=12)
    args=parser.parse_args()
    if min(args.resolution,args.count,args.cond_channels)<1:parser.error('Values must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);records=[]
    for index in range(args.count):
        grid=np.zeros((args.resolution,)*3,dtype=bool)
        a=rng.integers(0,max(1,args.resolution//2),size=3)
        b=rng.integers(max(1,args.resolution//2),args.resolution+1,size=3)
        grid[tuple(slice(lo,hi) for lo,hi in zip(a,b))]=True
        np.savez_compressed(output/f'{index}.npz',occupancy=grid,
                            cond=rng.normal(size=(2,5,args.cond_channels)).astype('float32'))
        records.append(dict(id=str(index),occupancy=f'{index}.npz',cond=f'{index}.npz'))
    (output/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    print(output/'manifest.jsonl')


if __name__=='__main__':main()

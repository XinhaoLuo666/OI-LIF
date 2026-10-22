"""Prepare SHD event-count frames or the historical SSC binary frames."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def binary_frames(times, units, time_steps=250):
    # Original code consumed events with time <= i*dt, discarding unit 0,
    # reversing channels as 700-unit, and dropping times > (T-1)*dt.
    times, units = np.asarray(times), np.asarray(units, dtype=np.int64)
    dt = 1.0 / time_steps
    boundaries = (np.arange(time_steps) * dt).astype(times.dtype)
    bins = np.searchsorted(boundaries, times, side="left")
    valid = (times >= 0) & (bins < time_steps) & (units > 0) & (units < 700)
    frames = np.zeros((time_steps, 700), dtype=np.uint8)
    frames[bins[valid], 700 - units[valid]] = 1
    return frames


def number_frames(units, time_steps=250):
    """SHD split_by=number: equal event counts, remainder in the final frame."""
    units=np.asarray(units,dtype=np.int64)
    if (units<0).any() or (units>=700).any():raise ValueError('SHD unit must be 0..699')
    frames=np.zeros((time_steps,700),dtype=np.float32);width=len(units)//time_steps
    for t in range(time_steps):
        left=t*width;right=(t+1)*width if t+1<time_steps else len(units)
        frames[t]=np.bincount(units[left:right],minlength=700)
    return frames

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',choices=['shd','ssc'],required=True)
    p.add_argument('--input',type=Path,required=True,help='Directory containing extracted HDF5 files.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time-steps',type=int,default=250)
    args=p.parse_args();splits=['train','test'] if args.dataset=='shd' else ['train','valid','test']
    for split in splits:
        with h5py.File(args.input/f'{args.dataset}_{split}.h5','r') as data:
            for i,label in enumerate(tqdm(data['labels'],desc=split)):
                if args.dataset=='shd':
                    target=args.output/split/str(int(label))/f'{i}.npz'
                    frames=number_frames(data['spikes/units'][i],args.time_steps)
                else:
                    target=args.output/split/f'sample_{i:06d}_{int(label)}.npy'
                    frames=binary_frames(data['spikes/times'][i],data['spikes/units'][i],args.time_steps)
                if target.exists():raise FileExistsError(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                if args.dataset=='shd':np.savez(target,frames=frames)
                else:np.save(target,frames,allow_pickle=False)

if __name__=='__main__':main()

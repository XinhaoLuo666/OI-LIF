"""Prepared SHD (NPZ event-count frames) and SSC (NPY binary frames)."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
class AudioFrames(Dataset):
    def __init__(self,root,split='train',dataset='shd',time_steps=250):
        root=Path(root)/split;self.T=time_steps;self.dataset=dataset
        self.files=sorted(root.glob('*/*.npz')) if dataset=='shd' else sorted(root.glob('*.npy'))
        if not self.files:raise FileNotFoundError(f'No prepared {dataset} data in {root}')
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        path=self.files[index]
        if self.dataset=='shd':
            with np.load(path,allow_pickle=False) as item:x=item['frames'].astype(np.float32)
            label=int(path.parent.name)
        else:x=np.load(path,allow_pickle=False).astype(np.float32);label=int(path.stem.split('_')[-1])
        if x.shape!=(self.T,700):raise ValueError(f'{path}: expected [{self.T},700], got {x.shape}')
        return torch.from_numpy(x),label

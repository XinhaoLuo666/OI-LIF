"""HARDVS grayscale event-frame reader, with experiment-specific polarity rules."""
from pathlib import Path
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HarDVS(Dataset):
    def __init__(self,root,split='train',time_steps=8,image_size=224,limit=0,variant='sdtv3',seed=42):
        self.T,self.size,self.variant,self.seed,self.split=time_steps,image_size,variant,seed,split
        root=Path(root);self.samples=[]
        for line in (root/'list'/f'{split}_label.txt').read_text().splitlines():
            name,_,label=line.split();parent=root/'rawframes'/name
            children=sorted(p for p in parent.iterdir() if p.is_dir())
            for clip in children or [parent]:
                self.samples.append((clip,int(label)))
                if limit and len(self.samples)>=limit:break
            if limit and len(self.samples)>=limit:break
        if limit:self.samples=self.samples[:limit]
        if not self.samples or any(not 0<=label<300 for _,label in self.samples):raise ValueError('Expected nonempty HARDVS samples with labels 0..299.')
        operations=[transforms.Resize(round(image_size*256/224),interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(image_size)]
        if split=='train':operations.append(transforms.RandomHorizontalFlip())
        self.transform=transforms.Compose(operations+[transforms.ToTensor()])
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        folder,label=self.samples[index]
        paths=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        if not paths:raise FileNotFoundError(folder)
        rng=random if self.split=='train' else random.Random(self.seed+index)
        start=rng.randint(0,max(0,len(paths)-self.T))
        frames=torch.zeros(self.T,2,self.size,self.size)
        for t,path in enumerate(paths[start:start+self.T]):
            with Image.open(path) as image:gray=self.transform(image.convert('L'))[0]
            if self.variant=='sdtv3':frames[t,0]=gray<.12;frames[t,1]=(gray>.3)&(gray<=.78)
            else:frames[t,0]=gray<.18;frames[t,1]=(gray>.3)&(gray<=.5)
        return frames,label

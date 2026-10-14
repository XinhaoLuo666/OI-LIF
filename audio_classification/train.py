"""Train or evaluate the legacy residual MLP with OI-LIF on SHD / SSC."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from classification import run
from runtime import common_arguments,device_from_arg,evaluation_recipe,loader,seed_everything
from audio_classification.dataset import AudioFrames
from audio_classification.model import AudioNet

def main():
    p=common_arguments(argparse.ArgumentParser(__doc__),epochs=20,batch_size=256,lr=.01,final_offset=2.,ramp_epochs=13)
    p.add_argument('--dataset',choices=['shd','ssc'],default='shd')
    p.add_argument('--channels',type=int,default=256);p.add_argument('--time-steps',type=int,default=250)
    p.add_argument('--selection-split',choices=['valid','test'],default='test');p.add_argument('--select-after',type=int,default=0)
    p.add_argument('--smoothing',type=float,default=0.);p.add_argument('--clip-grad',type=float,default=0.)
    args=evaluation_recipe(p.parse_args());seed_everything(args.seed)
    model=AudioNet(args.channels,num_classes=20 if args.dataset=='shd' else 35,residual='post',max_spikes=args.max_spikes).to(device_from_arg(args.device))
    def make(split,train=False):return loader(AudioFrames(args.data,split,args.dataset,args.time_steps),args,train=train)
    selection=make(args.selection_split);test=selection if args.selection_split=='test' else make('test')
    if args.eval:run(args,model,None,selection,test,None,None);return
    train=make('train',True)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.0001)
    run(args,model,train,selection,test,optimizer,scheduler)
if __name__=='__main__':main()

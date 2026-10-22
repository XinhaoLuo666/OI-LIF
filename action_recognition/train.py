"""Train or evaluate SDTv3 and ResNet34 with OI-LIF on HARDVS."""
import argparse,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from timm.optim import Lamb
from classification import run
from runtime import common_arguments,device_from_arg,evaluation_recipe,loader,seed_everything,load_checkpoint
from action_recognition.dataset import HarDVS
from action_recognition.sdtv3 import SDTv3
from action_recognition.resnet34 import ResNet34

def main():
    p=common_arguments(argparse.ArgumentParser(__doc__),epochs=100,batch_size=24,lr=1.5e-4,final_offset=1.,ramp_epochs=66)
    p.add_argument('--model',choices=['sdtv3','resnet34'],default='sdtv3')
    p.add_argument('--time-steps',type=int,default=8);p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--warmup-epochs',type=int,default=10);p.add_argument('--smoothing',type=float,default=.1)
    p.add_argument('--clip-grad',type=float,default=1.);p.add_argument('--drop-path',type=float,default=.1)
    p.add_argument('--accumulation',type=int,default=8);p.add_argument('--ema-decay',type=float,default=.99985)
    p.add_argument('--selection-split',choices=['val','test'],default='test');p.add_argument('--select-after',type=int,default=0)
    p.add_argument('--init',type=Path,help='Optional initial state_dict / pretraining checkpoint.')
    args=evaluation_recipe(p.parse_args());seed_everything(args.seed)
    model=(SDTv3(drop_path=args.drop_path) if args.model=='sdtv3' else ResNet34(learnable_tau=False,reset_mode='hard',detach_reset=False))
    if args.init:
        ck=load_checkpoint(args.init);state=ck.get('model',ck.get('model_state_dict',ck))
        state={k:v for k,v in state.items() if not k.startswith('head.') or k not in model.state_dict() or v.shape==model.state_dict()[k].shape}
        result=model.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not (k.endswith('.offset') or k.startswith('head.')) for k in result.missing_keys):raise ValueError(result)
    device=device_from_arg(args.device);model.to(device)
    limit=args.batch_size*args.max_batches
    def make(split,train=False):
        return loader(HarDVS(args.data,split,args.time_steps,args.image_size,limit,variant=args.model,seed=args.seed),args,train=train,drop_last=train)
    if args.eval:
        split=args.selection_split;selection=make(split)
        run(args,model,None,selection,selection,None,None);return
    train=make('train',True);selection=make(args.selection_split)
    test=selection if args.selection_split=='test' else make('test')
    groups=[dict(params=[v for v in model.parameters() if v.ndim>1],weight_decay=args.weight_decay),dict(params=[v for v in model.parameters() if v.ndim<=1],weight_decay=0.)]
    optimizer=Lamb(groups,lr=args.lr,trust_clip=True)
    steps=min(len(train),args.max_batches) if args.max_batches else len(train)
    def lr_factor(step):
        epoch=step/steps
        if args.warmup_epochs and epoch<args.warmup_epochs:return epoch/args.warmup_epochs
        progress=min(max((epoch-args.warmup_epochs)/max(args.epochs-args.warmup_epochs,1),0),1)
        return args.min_lr/args.lr+(1-args.min_lr/args.lr)*(1+math.cos(math.pi*progress))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    run(args,model,train,selection,test,optimizer,scheduler,scheduler_each_batch=True)
if __name__=='__main__':main()

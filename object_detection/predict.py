"""Predict boxes on one prepared event-frame NPY file."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from runtime import device_from_arg, load_checkpoint, load_model_weights
from oilif import SpikeCounter
from object_detection.model import SpikeYOLO
from object_detection.dataset import Gen1
from object_detection.evaluate import nms, native_boxes

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('prediction.json'));p.add_argument('--device',default='cuda:0');p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--half',action=argparse.BooleanOptionalAction,default=None)
    args=p.parse_args();device=device_from_arg(args.device);ck=load_checkpoint(args.weights)
    recipe=ck['args'];model=SpikeYOLO(ck['model_config']).to(device).eval();load_model_weights(model,ck,recipe['final_offset'])
    half=recipe.get('half',False) if args.half is None else args.half
    if half:
        if device.type!='cuda':raise ValueError('FP16 inference requires CUDA; use --no-half for FP32.')
        model.half()
    data=Gen1(args.source.parent,'',image_size=recipe['image_size'],layout=recipe['layout'],reverse_time=not recipe['chronological'],time_steps=recipe['time_steps'],num_classes=ck['model_config']['nc'],rectangular=True,allow_missing_labels=True,return_geometry=True,files=[args.source])
    image,_,geometry=data[0];counter=SpikeCounter(model)
    with torch.inference_mode():predictions,_=model(image.unsqueeze(0).to(device=device,dtype=next(model.parameters()).dtype));boxes=nms(predictions,confidence=args.confidence)[0]
    boxes=boxes.clone();boxes[:,:4]=native_boxes(boxes[:,:4],geometry.to(device))
    result=dict(source=args.source.name,columns=['x1','y1','x2','y2','confidence','class_id'],boxes=boxes.cpu().tolist(),**counter.result());counter.close()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

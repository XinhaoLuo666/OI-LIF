# OILIF

PyTorch code for action recognition, event-based object detection and audio classification.

## Main results

| Model | Dataset | Accuracy / mAP50:95 (%) | Energy (mJ) | Firing rate |
|---|---|---:|---:|---:|
| [SDTv3](action_recognition/logs/sdtv3_hardvs.txt) | HARDVS | 49.8 | 5.5 | — |
| ResNet34 | HARDVS | 48.5 | 1.9 | — |
| [SpikeYOLO](object_detection/logs/yolo_gen1.txt) | GEN1 | 46.7 | 7.2 | — |
| [SpikeYOLO](object_detection/logs/yolo_dsec_det.txt) | DSEC-DET | 24.9 | 5.3 | — |
| [SpikeYOLO](object_detection/logs/yolo_pku_davis_sod.txt) | PKU-DAVIS-SOD | 23.7 | 6.5 | — |
| [Residual MLP](audio_classification/logs/shd.txt) | SHD | 75.73 ± 0.93 | — | 0.171 ± 0.006 |
| [Residual MLP](audio_classification/logs/ssc.txt) | SSC | 63.40 ± 0.13 | — | 0.123 ± 0.003 |

Results and estimated energy follow manuscript v12, Tables 1–3. Audio tasks report firing rate; SHD/SSC values are five-seed mean ± sample standard deviation. Included logs each represent one run. The ResNet34 log is not included.

Weights are not included. The archived GEN1 checkpoint was verified at 46.657% mAP50:95 on all 30,605 test samples; see [verification](gen1_validation.json).

## Setup

```bash
git clone https://github.com/XinhaoLuo666/OI-LIF.git
cd OI-LIF
conda create -n oilif python=3.11 -y
conda activate oilif
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
nvidia-smi
python check.py --device cuda:0
```

## Training

### HARDVS

Data: `list/{train,test}_label.txt` and `rawframes/<clip>/<segment>/frame*.png`.

```bash
# SDTv3: vth=2, log ramp; add --init /path/to/initial_sfa_checkpoint.pth for historical initialization
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --final-offset 2 --seed 42 --output runs/sdtv3 --amp

# ResNet34: vth=1, cosine ramp
python action_recognition/train.py --data /path/to/HARDVS --model resnet34 \
  --epochs 300 --batch-size 48 --accumulation 4 --lr 0.0003 --weight-decay 0.06 \
  --schedule cosine --ramp-epochs 200 --final-offset 1 --seed 0 --ema-decay 0 \
  --output runs/resnet34 --amp
```

### Object detection

Data: `{train,test}/*.npy` event frames `[4,W,H,3]` with adjacent normalized YOLO `.txt` labels.

```bash
# Optional GEN1 preparation
python object_detection/prepare.py --input /path/to/gen1_raw --output /path/to/gen1_frames

python object_detection/train.py --data /path/to/gen1_frames \
  --batch-size 140 --final-offset 2 --output runs/gen1 --amp
python object_detection/train.py --data /path/to/dsec_frames --epochs 100 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 2 \
  --schedule log --ramp-epochs 66 --final-offset 2 --output runs/dsec --amp
python object_detection/train.py --data /path/to/pku_frames --epochs 200 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 3 \
  --schedule log --ramp-epochs 133 --final-offset 2 --output runs/pku --amp
```

### SHD / SSC

```bash
# Optional preparation from official HDF5 files
python audio_classification/prepare.py --dataset shd --input /path/to/shd_h5 --output /path/to/shd_frames
python audio_classification/prepare.py --dataset ssc --input /path/to/ssc_h5 --output /path/to/ssc_frames

python audio_classification/train.py --dataset shd --data /path/to/shd_frames \
  --seed 41 --final-offset 2 --output runs/shd
python audio_classification/train.py --dataset ssc --data /path/to/ssc_frames \
  --seed 40 --final-offset 2 --output runs/ssc
```

## Evaluation and inference

```bash
# Trained SDTv3
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --eval runs/sdtv3/best.pt --output runs/sdtv3_eval

# Trained ResNet34
python action_recognition/train.py --data /path/to/HARDVS --model resnet34 \
  --eval runs/resnet34/best.pt --output runs/resnet34_eval

# Trained GEN1; use runs/dsec/best.pt or runs/pku/best.pt for other datasets
python object_detection/train.py --data /path/to/gen1_frames \
  --eval runs/gen1/best.pt --batch-size 140 --half --output runs/gen1_eval
python object_detection/predict.py --weights runs/gen1/best.pt \
  --source /path/to/sample.npy --half --output prediction.json --confidence 0.25

python audio_classification/train.py --dataset shd --data /path/to/shd_frames \
  --eval runs/shd/best.pt --output runs/shd_eval
python audio_classification/train.py --dataset ssc --data /path/to/ssc_frames \
  --eval runs/ssc/best.pt --output runs/ssc_eval
```

Resume with the same training arguments plus `--resume runs/<experiment>/last.pt`.
Training writes `best.pt`, `last.pt` and `history.jsonl`; evaluation writes `evaluation.json`.

Historical recipes and provenance: [experiments.json](experiments.json). Short GPU training checks: [training_validation.json](training_validation.json); full training has not been rerun. Licenses: [NOTICE](NOTICE).

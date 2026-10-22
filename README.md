# OI-LIF

PyTorch code for event-based action recognition, object detection, and audio classification.

## Main results

### Action recognition and object detection

| Model | Dataset | Metric | Result (%) | Energy (mJ) |
|---|---|---|---:|---:|
| SDTv3 (E-SpikeFormer) | HARDVS | Accuracy | 49.8 | 5.5 |
| ResNet34 | HARDVS | Accuracy | 48.5 | 1.9 |
| SpikeYOLO | GEN1 | mAP50:95 | 46.7 | 7.2 |
| SpikeYOLO | DSEC-DET | mAP50:95 | 24.9 | 5.3 |
| SpikeYOLO | PKU-DAVIS-SOD | mAP50:95 | 23.7 | 6.5 |

### Audio classification

| Model | Dataset | Accuracy (%) | Firing rate |
|---|---|---:|---:|
| Residual MLP | SHD | 75.73 ± 0.93 | 0.171 ± 0.006 |
| Residual MLP | SSC | 63.40 ± 0.13 | 0.123 ± 0.003 |

## Weights

| Model / dataset | Download | Save as |
|---|---|---|
| SDTv3 / HARDVS | [Google Drive](https://drive.google.com/file/d/1tp1LHjtYSzocwcHWF-hXIKAFDCNYvNhT/view?usp=sharing) | `action_recognition/weights/sdtv3_hardvs_oilif.pt` |
| SpikeYOLO / GEN1 | [Google Drive](https://drive.google.com/file/d/1d36GJVnq7FR4JTGF9p0K-BusdVaVO_28/view?usp=sharing) | `object_detection/weights/gen1_oilif.pt` |

```bash
mkdir -p action_recognition/weights object_detection/weights
```

## Code structure

```text
OI-LIF/
├── oilif.py                 # Shared neurons and offset schedules
├── runtime.py               # Data loading and checkpoint utilities
├── classification.py        # Shared classification training and evaluation
├── action_recognition/      # SDTv3 and ResNet34
├── object_detection/        # SpikeYOLO
├── audio_classification/    # Residual MLP for SHD and SSC
└── requirements.txt
```

## Training

Run from the repository root and replace `/path/to/...` with your dataset paths.

### HARDVS

```bash
# SDTv3
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --final-offset 2 --seed 42 --output runs/sdtv3 --amp

# ResNet34
python action_recognition/train.py --data /path/to/HARDVS --model resnet34 \
  --epochs 300 --batch-size 48 --accumulation 4 --lr 0.0003 --weight-decay 0.06 \
  --schedule cosine --ramp-epochs 200 --final-offset 1 --seed 0 --ema-decay 0 \
  --output runs/resnet34 --amp
```

### Object detection

```bash
# GEN1
python object_detection/train.py --data /path/to/gen1_frames \
  --batch-size 140 --final-offset 2 --output runs/gen1 --amp

# DSEC-DET
python object_detection/train.py --data /path/to/dsec_frames --epochs 100 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 2 \
  --schedule log --ramp-epochs 66 --final-offset 2 --output runs/dsec --amp

# PKU-DAVIS-SOD
python object_detection/train.py --data /path/to/pku_frames --epochs 200 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 3 \
  --schedule log --ramp-epochs 133 --final-offset 2 --output runs/pku --amp
```

### SHD / SSC

```bash
python audio_classification/train.py --dataset shd --data /path/to/shd_frames \
  --seed 41 --final-offset 2 --output runs/shd
python audio_classification/train.py --dataset ssc --data /path/to/ssc_frames \
  --seed 40 --final-offset 2 --output runs/ssc
```

## Evaluation and inference

### Downloaded weights

```bash
# HARDVS
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --eval action_recognition/weights/sdtv3_hardvs_oilif.pt --output runs/sdtv3_eval

# GEN1
python object_detection/train.py --data /path/to/gen1_frames \
  --eval object_detection/weights/gen1_oilif.pt --batch-size 140 --half \
  --output runs/gen1_eval

# Detection on one event-frame file
python object_detection/predict.py --weights object_detection/weights/gen1_oilif.pt \
  --source /path/to/sample.npy --half --confidence 0.25 --output prediction.json
```

### Trained weights

```bash
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --eval runs/sdtv3/best.pt --output runs/sdtv3_eval
python action_recognition/train.py --data /path/to/HARDVS --model resnet34 \
  --eval runs/resnet34/best.pt --output runs/resnet34_eval

python object_detection/train.py --data /path/to/gen1_frames \
  --eval runs/gen1/best.pt --batch-size 140 --half --output runs/gen1_eval
python object_detection/train.py --data /path/to/dsec_frames \
  --eval runs/dsec/best.pt --half --output runs/dsec_eval
python object_detection/train.py --data /path/to/pku_frames \
  --eval runs/pku/best.pt --half --output runs/pku_eval

python audio_classification/train.py --dataset shd --data /path/to/shd_frames \
  --eval runs/shd/best.pt --output runs/shd_eval
python audio_classification/train.py --dataset ssc --data /path/to/ssc_frames \
  --eval runs/ssc/best.pt --output runs/ssc_eval
```

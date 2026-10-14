# OI-LIF

PyTorch implementation of OI-LIF for event-based action recognition, object detection, and audio classification. This repository provides training, evaluation, and detection inference code, together with selected experiment logs and verification reports.

| Task | Models | Datasets |
|---|---|---|
| Action recognition | SDTv3, spiking ResNet34 | HARDVS |
| Object detection | SpikeYOLO | GEN1, DSEC-DET, PKU-DAVIS-SOD |
| Audio classification | Residual MLP | SHD, SSC |

Datasets and model weights are obtained separately. Weight files, dataset directories, and generated training runs are excluded from Git.

## Main results

The following values are reported in manuscript Tables 1–3 and recorded in [manuscript_alignment.json](manuscript_alignment.json).

### Action recognition and object detection

| Model | Dataset | Metric | Result (%) | Estimated energy (mJ) |
|---|---|---|---:|---:|
| [SDTv3](action_recognition/logs/sdtv3_hardvs.txt) | HARDVS | Accuracy | 49.8 | 5.5 |
| ResNet34 | HARDVS | Accuracy | 48.5 | 1.9 |
| [SpikeYOLO](object_detection/logs/yolo_gen1.txt) | GEN1 | mAP50:95 | 46.7 | 7.2 |
| [SpikeYOLO](object_detection/logs/yolo_dsec_det.txt) | DSEC-DET | mAP50:95 | 24.9 | 5.3 |
| [SpikeYOLO](object_detection/logs/yolo_pku_davis_sod.txt) | PKU-DAVIS-SOD | mAP50:95 | 23.7 | 6.5 |

The log corresponding to the manuscript's ResNet34 result is not included.

### Audio classification

| Model | Dataset | Accuracy (%) | Firing rate |
|---|---|---:|---:|
| [Residual MLP](audio_classification/logs/shd.txt) | SHD | 75.73 ± 0.93 | 0.171 ± 0.006 |
| [Residual MLP](audio_classification/logs/ssc.txt) | SSC | 63.40 ± 0.13 | 0.123 ± 0.003 |

SHD and SSC values are the mean ± sample standard deviation over seeds 40–44; each included audio log represents one run.

The archived GEN1 checkpoint was evaluated with this code on all **30,605 test samples**, obtaining **46.657% mAP50:95** and **74.957% mAP50**. See [gen1_validation.json](gen1_validation.json) for the protocol and comparison with the original implementation. Full training runs have not been repeated for this release.

## Installation

The verified environment uses Python 3.11, PyTorch 2.1.0, torchvision 0.16.0, and CUDA 12.1. Run the commands below from the repository root.

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

The code does not require a modified SpikingJelly or Ultralytics installation, or CuPy. Training and inference default to `cuda:0`; use `--device cuda:<index>` to select another visible GPU. Each training entry point uses one GPU.

## Repository structure

```text
OI-LIF/
├── oilif.py                 # Integer firing, neuron state, and offset schedules
├── runtime.py               # Devices, data loaders, and checkpoint handling
├── classification.py        # Shared action/audio training and evaluation
├── check.py                 # Neuron and model forward/backward checks
├── action_recognition/      # HARDVS readers, SDTv3, and ResNet34
├── object_detection/        # SpikeYOLO, preprocessing, loss, evaluation, inference
├── audio_classification/    # SHD/SSC preprocessing and residual MLP
├── requirements.txt
├── experiments.json         # Historical recipes and experiment provenance
├── weights.json             # Archived checkpoint paths, sizes, and SHA-256 hashes
└── *_validation.json        # Recorded training and GEN1 evaluation checks
```

## Checkpoints

Model weights are not stored in this GitHub repository. [weights.json](weights.json) documents the two archived checkpoints and their SHA-256 hashes. Download the files below, then rename and place them at the listed paths to use the archived-checkpoint examples:

| Model / dataset | Download | Checkpoint path relative to the repository root | Final offset |
|---|---|---|---:|
| SDTv3 / HARDVS | [Google Drive](https://drive.google.com/file/d/1tp1LHjtYSzocwcHWF-hXIKAFDCNYvNhT/view?usp=sharing) | `action_recognition/weights/sdtv3_hardvs_oilif.pt` | 2 |
| SpikeYOLO / GEN1 | [Google Drive](https://drive.google.com/file/d/1d36GJVnq7FR4JTGF9p0K-BusdVaVO_28/view?usp=sharing) | `object_detection/weights/gen1_oilif.pt` | 2 |

```bash
mkdir -p action_recognition/weights object_detection/weights
```

The SDTv3 archive contains the epoch-99 model state (49.738% recorded accuracy). The log's 49.799% peak occurred at epoch 94 and was not retained as a separate checkpoint. The historical SDTv3 recipe also used a separate SFA initialization checkpoint.

## Data preparation

### HARDVS

```text
HARDVS/
├── list/
│   ├── train_label.txt
│   └── test_label.txt
└── rawframes/
    └── <clip>/
        └── <segment>/
            └── frame*.png
```

Each label-file row has three whitespace-separated fields: `<clip> <frame_count> <class_id>`, with zero-based class IDs from 0 to 299. Frames can also be placed directly inside `<clip>`. The reader applies model-specific polarity rules and uses eight frames at 224 × 224 by default.

### GEN1, DSEC-DET, and PKU-DAVIS-SOD

```text
dataset_frames/
├── train/
│   ├── sample.npy
│   └── sample.txt
└── test/
    ├── sample.npy
    └── sample.txt
```

The default event-frame layout is `[4, W, H, 3]` (`--layout whc`). Each label row contains `class_id x_center y_center width height`, with normalized coordinates referring to the physical sensor image. A split can alternatively contain `images/*.npy` and matching `labels/*.txt`. Use `--layout hwc` for `[4, H, W, 3]` arrays. Time is reversed by default to match the archived experiments; `--chronological` disables this reversal.

For raw GEN1 data, the converter reads paired `*_td.dat` event files and `*_bbox.npy` annotations:

```bash
python object_detection/prepare.py \
  --input /path/to/gen1_raw --output /path/to/gen1_frames \
  --splits train test
```

It generates four frames per sample over a 250,000 μs window. Prepare DSEC-DET and PKU-DAVIS-SOD separately in the reader's frame/label format; the included DAT converter is specific to GEN1.

### SHD and SSC

The input directory must contain extracted `shd_train.h5` and `shd_test.h5`, or `ssc_train.h5`, `ssc_valid.h5`, and `ssc_test.h5`.

```bash
python audio_classification/prepare.py --dataset shd \
  --input /path/to/shd_h5 --output /path/to/shd_frames
python audio_classification/prepare.py --dataset ssc \
  --input /path/to/ssc_h5 --output /path/to/ssc_frames
```

SHD output uses `<split>/<class_id>/<sample>.npz` with a `frames` array. SSC output uses `<split>/sample_<index>_<class_id>.npy`. Both have shape `[250, 700]`. SHD uses event-count frames split by event number; SSC retains the historical binary-frame representation, channel ordering, and boundary rules.

## Training

Replace `/path/to/...` with your dataset locations. The examples use the selected archived recipe settings; details are recorded in [experiments.json](experiments.json).

### Action recognition

```bash
# SDTv3: 100 epochs, final offset 2, logarithmic ramp over 66 epochs.
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --final-offset 2 --seed 42 --output runs/sdtv3 --amp

# ResNet34: 300 epochs, final offset 1, cosine ramp over 200 epochs.
python action_recognition/train.py --data /path/to/HARDVS --model resnet34 \
  --epochs 300 --batch-size 48 --accumulation 4 --lr 0.0003 --weight-decay 0.06 \
  --schedule cosine --ramp-epochs 200 --final-offset 1 --seed 0 --ema-decay 0 \
  --output runs/resnet34 --amp
```

For the historical SDTv3 initialization, add `--init /path/to/initial_sfa_checkpoint.pth`; without it, training starts from random initialization. Historical action-recognition runs used multiple GPUs, while these entry points use one GPU with gradient accumulation.

### Object detection

```bash
# GEN1: 200 epochs, two classes, cosine offset ramp.
python object_detection/train.py --data /path/to/gen1_frames \
  --batch-size 140 --final-offset 2 --output runs/gen1 --amp

# DSEC-DET: 100 epochs, two classes, logarithmic offset ramp.
python object_detection/train.py --data /path/to/dsec_frames --epochs 100 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 2 \
  --schedule log --ramp-epochs 66 --final-offset 2 --output runs/dsec --amp

# PKU-DAVIS-SOD: 200 epochs, three classes, logarithmic offset ramp.
python object_detection/train.py --data /path/to/pku_frames --epochs 200 \
  --batch-size 160 --image-size 320 --lr 0.01 --num-classes 3 \
  --schedule log --ramp-epochs 133 --final-offset 2 --output runs/pku --amp
```

### Audio classification

```bash
python audio_classification/train.py --dataset shd --data /path/to/shd_frames \
  --seed 41 --final-offset 2 --output runs/shd
python audio_classification/train.py --dataset ssc --data /path/to/ssc_frames \
  --seed 40 --final-offset 2 --output runs/ssc
```

Both audio recipes use 20 epochs, batch size 256, learning rate 0.01, and a logarithmic offset ramp over 13 epochs. For five-seed experiments, repeat with `--seed 40` through `--seed 44` and separate output directories.

Training defaults to selecting the best checkpoint on the test split to match the archived experiments. To select on an independent validation split, provide that split and use `--selection-split val` for action recognition or detection, or `--selection-split valid` for SSC.

## Evaluation and inference

### Archived checkpoints

After obtaining and placing the checkpoints described above:

```bash
python action_recognition/train.py --data /path/to/HARDVS --model sdtv3 \
  --eval action_recognition/weights/sdtv3_hardvs_oilif.pt --output runs/sdtv3_archive_eval

python object_detection/train.py --data /path/to/gen1_frames \
  --eval object_detection/weights/gen1_oilif.pt --batch-size 140 --half \
  --output runs/gen1_archive_eval

python object_detection/predict.py --weights object_detection/weights/gen1_oilif.pt \
  --source /path/to/sample.npy --half --confidence 0.25 --output prediction.json
```

The GEN1 verification used FP16 evaluation; `--no-half` selects FP32. Inference accepts a single event-frame `.npy` without a label file and writes sensor-coordinate boxes with columns `x1, y1, x2, y2, confidence, class_id`, together with firing-rate statistics.

### Your own trained checkpoints

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

Evaluation restores the saved model recipe from the checkpoint. Detection evaluates the test split by default; use `--eval-split val` for its validation split.

## Resuming and outputs

Resume with the original training arguments plus `--resume runs/<experiment>/last.pt`. For example:

```bash
python object_detection/train.py --data /path/to/gen1_frames \
  --batch-size 140 --final-offset 2 --output runs/gen1 --amp \
  --resume runs/gen1/last.pt
```

| Output | Contents |
|---|---|
| `config.json` | Training arguments |
| `history.jsonl` | Epoch losses, metrics, and offset values |
| `best.pt` | Best selected checkpoint |
| `last.pt` | Latest checkpoint for resuming training |
| `evaluation.json` | Evaluation metrics |

Training checkpoints retain model parameters, optimizer and scheduler state, AMP state, and random-number-generator state. Generated outputs stay in the ignored `runs/` directory.

## OI-LIF implementation

[oilif.py](oilif.py) implements integer firing, a straight-through gradient, and scheduled additive offsets. The default neuron accepts time-major input `[T, B, ...]`, with `tau=2`, maximum integer output `D=4`, and soft reset:

```text
u[t] = v[t-1] + (x[t] - v[t-1]) / tau
q[t] = round(clamp(u[t] - offset, 0, D))
v[t] = u[t] - stop_gradient(q[t])
```

The firing gradient is 1 inside `[0, D]` and 0 outside. The offset ramps from zero to `--final-offset` over `--ramp-epochs`, using `log10(1 + 9p)` or `(1 - cos(pi * p)) / 2`, with progress `p` clipped to `[0, 1]`.

ResNet34 retains fixed `tau=2`, hard reset, and `detach_reset=False`. SDTv3 uses `tau=None` with a single IF activation over the complete time tensor. Call `reset_net(model)` between independent batches; reset clears membrane state while preserving the checkpointed offset.

## Verification and provenance

| Report | Scope |
|---|---|
| [experiments.json](experiments.json) | Historical recipes, log provenance, and five-seed audio statistics |
| [weights.json](weights.json) | Archived checkpoint metadata and hashes |
| [verification.json](verification.json) | Recorded implementation and model checks |
| [training_validation.json](training_validation.json) | Short GPU training, parameter updates, resume checks, and checkpoint evaluation |
| [gen1_validation.json](gen1_validation.json) | Full GEN1 test evaluation and parity with the original implementation |
| [gen1_neuron_audit.json](gen1_neuron_audit.json) | GEN1 neuron offsets and integer firing limits |
| [manuscript_alignment.json](manuscript_alignment.json) | Correspondence between reported results and archived evidence |

Recorded short training checks passed for SDTv3, ResNet34, SHD, SSC, GEN1, and DSEC-DET. PKU-DAVIS-SOD data were unavailable for its short training check. These reports validate execution and checkpoint behavior; they do not represent full retraining of the reported experiments.

## Licenses and acknowledgments

See [NOTICE](NOTICE) for architecture sources and retained third-party terms. The root [LICENSE](LICENSE) and detection code retain AGPL-3.0 terms; additional license texts are in [LICENSE-SPIKINGJELLY](LICENSE-SPIKINGJELLY) and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Dataset licenses apply separately.

"""Small shared utilities for reproducible training and checkpointing."""
import json
import os
import random
import pickle
import re
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oilif import reset_net, set_offset


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def device_from_arg(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and GPU access.")
    return device


def loader(dataset, args, train=False, collate_fn=None, drop_last=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, pin_memory=True,
                      worker_init_fn=seed_worker, generator=generator,
                      collate_fn=collate_fn, drop_last=drop_last)


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch,
                    best, generator, **extra):
    reset_net(model)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                 args=vars(args), epoch=epoch, best=best,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                          loader=generator.get_state()), **extra)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def evaluation_recipe(args):
    """Rebuild the saved architecture/recipe; retain current IO/device flags."""
    if args.eval:
        saved = load_checkpoint(args.eval).get("args", {})
        for key, value in saved.items():
            if key == "half" and getattr(args, key, None) is not None:
                continue
            if key not in {"eval", "eval_split", "resume", "init", "data", "output", "device", "workers", "max_batches", "batch_size"}:
                setattr(args, key, value)
    return args


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def common_arguments(parser, *, epochs, batch_size, lr, final_offset, ramp_epochs):
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/experiment"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--final-offset", type=float, default=final_offset)
    parser.add_argument("--ramp-epochs", type=int, default=ramp_epochs)
    parser.add_argument("--schedule", choices=["log", "cosine"], default="log")
    parser.add_argument("--max-spikes", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit batches for a smoke test; 0 uses the entire split.")
    return parser


class _ArchivedModule(torch.nn.Module):
    """Carrier for reading historical module objects; only their state is used."""
    pass

class _ArchiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("ultralytics.", "spikingjelly.", "Qtrick_architecture.")):
            return _ArchivedModule
        return super().find_class(module, name)

_ARCHIVE_PICKLE = SimpleNamespace(__name__="pickle", Unpickler=_ArchiveUnpickler,
                                load=pickle.load, loads=pickle.loads)

def load_checkpoint(path):
    """Load the original archive in place; no weight conversion or rewriting.

    Legacy YOLO archives contain entire modules. Carrier classes allow extracting
    their state without shipping two large framework forks. Current models are
    reconstructed from their saved architecture, then load the unchanged tensors.
    """
    checkpoint = torch.load(path, map_location="cpu", pickle_module=_ARCHIVE_PICKLE, weights_only=False)
    if "model_state_dict" in checkpoint:
        checkpoint["model"] = checkpoint["model_state_dict"]
    if isinstance(checkpoint.get("model"), torch.nn.Module):
        original = checkpoint.get("ema")
        original = original if original is not None else checkpoint["model"]
        config = dict(original.yaml)
        checkpoint["model_config"] = config
        checkpoint["model"] = original.state_dict()
        checkpoint.pop("ema", None)
        raw = checkpoint.get("train_args", {})
        # Carrier classes do not implement SpikingJelly's dynamic memory
        # attributes. Read the persisted memory if metrics are unavailable.
        thresholds = [float(m._memories['v_threshold']) for m in original.modules()
                      if 'v_threshold' in getattr(m, '_memories', {})]
        offset = checkpoint.get("train_metrics", {}).get("threshold")
        if offset is None:
            offset = raw.get("final_offset", thresholds[0] if thresholds else 1.)
        checkpoint["args"] = dict(num_classes=config["nc"], time_steps=4,
            image_size=int(raw.get("imgsz",320)), layout="whc", chronological=False,
            max_spikes=4, final_offset=offset, half=True)
    elif not isinstance(checkpoint.get("args", {}), dict):
        # Original argparse.Namespace fields not used by these entry points are ignored.
        original = vars(checkpoint["args"])
        keys = ["epochs", "batch_size", "lr", "weight_decay", "min_lr", "warmup_epochs",
                "smoothing", "drop_path", "seed", "amp"]
        checkpoint["args"] = {k: original[k] for k in keys if k in original}
        if "downsample1_1.encode_conv.weight" in checkpoint["model"]:
            # These original SDTv3 archives do not store neuron thresholds.
            # Their saved run name distinguishes the final-vth=2 recipe even
            # when the unchanged archive is copied or renamed for release.
            run_name = str(original.get("output_dir", ""))
            offset = original.get("final_offset", 2. if re.search(r"_2th_", run_name) else 1.)
            checkpoint["args"].update(model="sdtv3", final_offset=float(offset), time_steps=8,
                image_size=224, max_spikes=4, schedule="log", ramp_epochs=66)
    return checkpoint

def load_model_weights(model, checkpoint, final_offset):
    """Accept absent offset buffers in historical files; reject architecture mismatch."""
    # Classification best.pt already stores the selected raw/EMA model in
    # "model". Its separate "ema" state is for resuming, and can have lower
    # validation accuracy. Detector checkpoints select their EMA instead.
    state = checkpoint["model"]
    if "model_config" in checkpoint and checkpoint.get("ema") is not None:
        state = checkpoint["ema"]
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith((".offset", ".stride"))]
    if missing or result.unexpected_keys:
        raise ValueError(f"Checkpoint architecture mismatch: {result}")
    if any(k.endswith(".offset") for k in result.missing_keys):
        set_offset(model, final_offset)
    return result

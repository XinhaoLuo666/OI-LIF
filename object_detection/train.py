"""Gen1 / SpikeYOLO training. Offsets are shared by training and EMA evaluation."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tqdm import tqdm

from oilif import set_offset, scheduled_offset
from runtime import (common_arguments, device_from_arg, evaluation_recipe, loader, restore_rng,
                     save_checkpoint, seed_everything, write_json, load_checkpoint, load_model_weights)
from object_detection.dataset import Gen1, collate
from object_detection.evaluate import evaluate
from object_detection.loss import v8DetectionLoss
from object_detection.model import SpikeYOLO


def main():
    parser = common_arguments(argparse.ArgumentParser(__doc__), epochs=200, batch_size=160,
                              lr=0.01, final_offset=2.0, ramp_epochs=133)
    parser.set_defaults(seed=0, schedule="cosine")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.yaml"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--layout", choices=["whc", "hwc"], default="whc")
    parser.add_argument("--chronological", action="store_true", help="Disable legacy time reversal.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None,
                        help="Use FP16 evaluation (default for historical YOLO archives).")
    parser.add_argument("--selection-split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--warmup-epochs", type=float, default=10)
    parser.add_argument("--lrf", type=float, default=0.004)
    args = evaluation_recipe(parser.parse_args())
    args.half = bool(args.half)
    if args.image_size % 32 or args.epochs < 1 or not 0 <= args.ramp_epochs <= args.epochs:
        raise ValueError("Use an image size divisible by 32 and 0 <= ramp_epochs <= epochs.")
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    checkpoint = load_checkpoint(args.eval or args.resume) if args.eval or args.resume else None
    import yaml
    config = checkpoint["model_config"] if checkpoint else yaml.safe_load(args.model.read_text())
    if not checkpoint:
        config["nc"] = args.num_classes
    model = SpikeYOLO(config, args.max_spikes).to(device)
    args.num_classes = config["nc"]
    args.output.mkdir(parents=True, exist_ok=True)
    def make_loader(split, train=False):
        dataset = Gen1(args.data, split, args.image_size, args.layout,
                       not args.chronological, args.time_steps, args.batch_size * args.max_batches, args.num_classes,
                       rectangular=not train, return_geometry=True)
        return loader(dataset, args, train, collate)
    selection = make_loader(args.eval_split if args.eval else args.selection_split)
    if args.eval:
        checkpoint = load_checkpoint(args.eval)
        load_model_weights(model, checkpoint, args.final_offset)
        if args.half:
            if device.type != "cuda":
                raise ValueError("FP16 evaluation requires a CUDA device; use --no-half for FP32.")
            model.half()
        result = evaluate(model, selection, device, args.max_batches)
        write_json(args.output / "evaluation.json", result)
        print(json.dumps(result))
        return
    train = make_loader("train", True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, norm, bias = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = bias if name.endswith(".bias") else norm if parameter.ndim == 1 else decay
        group.append(parameter)
    accumulate = max(round(64 / args.batch_size), 1)
    optimizer = torch.optim.SGD(bias, lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group(dict(params=decay, weight_decay=args.weight_decay * args.batch_size * accumulate / 64))
    optimizer.add_param_group(dict(params=norm, weight_decay=0.0))
    lf = lambda epoch: (1 - epoch / args.epochs) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = v8DetectionLoss(model)
    best, start, updates = -1.0, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        for key, value in checkpoint["args"].items():
            if key not in {"resume", "eval", "eval_split", "output", "device", "workers", "data", "model"}:
                if str(getattr(args, key, None)) != str(value):
                    raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], train.generator)
        start, best, updates = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["updates"]
    write_json(args.output / "config.json", {k: str(v) if hasattr(v, "__fspath__") else v
                                            for k, v in vars(args).items()})
    steps = min(len(train), args.max_batches) if args.max_batches else len(train)
    warmup_steps = max(round(args.warmup_epochs * steps), 100) if args.warmup_epochs else 0
    for epoch in range(start, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = torch.zeros(3, device=device)
        last_step = epoch * steps - 1
        for step, batch in enumerate(tqdm(train, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if step >= steps:
                break
            global_step = epoch * steps + step
            accumulation = accumulate
            if warmup_steps and global_step <= warmup_steps:
                accumulation = max(1, round(np.interp(global_step, [0, warmup_steps], [1, 64 / args.batch_size])))
                for i, group in enumerate(optimizer.param_groups):
                    group["lr"] = float(np.interp(global_step, [0, warmup_steps],
                                                  [0.1 if i == 0 else 0.0, args.lr * lf(epoch)]))
                    group["momentum"] = float(np.interp(global_step, [0, warmup_steps], [0.8, args.momentum]))
            set_offset(model, scheduled_offset(global_step + 1, args.ramp_epochs * steps,
                                              args.final_offset, args.schedule))
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=args.amp):
                loss, items = criterion(model(batch["img"]), batch)
            if not math.isfinite(loss.item()):
                raise FloatingPointError("Nonfinite detection loss.")
            scaler.scale(loss).backward()
            if global_step - last_step >= accumulation or step + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                last_step = global_step
                if scaler.get_scale() >= previous_scale:
                    updates += 1
                    weight = 0.9999 * (1 - math.exp(-updates / 2000))
                    with torch.no_grad():
                        current_state = model.state_dict()
                        for key, value in ema.state_dict().items():
                            current = current_state[key]
                            if value.is_floating_point() and not key.endswith(".offset"):
                                value.mul_(weight).add_(current, alpha=1 - weight)
                            else:
                                value.copy_(current)
            losses += items
        scheduler.step()
        offset = scheduled_offset((epoch + 1) * steps, args.ramp_epochs * steps,
                                  args.final_offset, args.schedule)
        set_offset(model, offset)
        set_offset(ema, offset)
        metrics = evaluate(ema, selection, device, args.max_batches)
        fitness = 0.1 * metrics["map50"] + 0.9 * metrics["map50_95"]
        improved = fitness > best
        best = max(best, fitness)
        row = dict(epoch=epoch, offset=offset, loss=(losses / steps).tolist(), **metrics)
        print(json.dumps(row))
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(args.output / "last.pt", model, optimizer, scheduler, scaler,
                        args, epoch, best, train.generator, ema=ema.state_dict(),
                        updates=updates, model_config=model.config)
        if improved:
            save_checkpoint(args.output / "best.pt", model, optimizer, scheduler, scaler,
                            args, epoch, best, train.generator, ema=ema.state_dict(),
                            updates=updates, model_config=model.config)
    if args.selection_split == "val" and (args.output / "best.pt").exists():
        checkpoint = torch.load(args.output / "best.pt", map_location="cpu")
        ema.load_state_dict(checkpoint["ema"])
        result = evaluate(ema, make_loader("test"), device, args.max_batches)
        write_json(args.output / "test.json", result)
        print("Best checkpoint test:", json.dumps(result))


if __name__ == "__main__":
    main()

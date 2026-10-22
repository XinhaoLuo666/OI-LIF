"""Training / evaluation shared by the two classification experiments."""
import json
import math
import copy

import torch
from tqdm import tqdm

from oilif import reset_net, scheduled_offset, set_offset, SpikeCounter
from runtime import device_from_arg, restore_rng, save_checkpoint, seed_everything, write_json, load_checkpoint, load_model_weights


@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=0):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    activity = SpikeCounter(model)
    for i, (x, y) in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device).reshape(-1)
        reset_net(model)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    reset_net(model)
    stats = activity.result(); activity.close()
    return dict(accuracy=100 * correct / total, loss=loss_sum / total, samples=total, **stats)


def run(args, model, train_loader, selection_loader, test_loader, optimizer, scheduler,
        scheduler_each_batch=False):
    if args.epochs < 1 or args.ramp_epochs < 0 or args.ramp_epochs > args.epochs:
        raise ValueError("Require epochs >= 1 and 0 <= ramp_epochs <= epochs.")
    device = device_from_arg(args.device)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.eval:
        checkpoint = load_checkpoint(args.eval)
        load_model_weights(model, checkpoint, args.final_offset)
        result = evaluate(model, test_loader, device, args.max_batches)
        print(json.dumps(result))
        write_json(args.output / "evaluation.json", result)
        return
    best, start = -1.0, 0
    accumulation = getattr(args, "accumulation", 1)
    decay = getattr(args, "ema_decay", 0.)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if decay else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # Exact resume requires the original recipe and batch order.
        for key, old in checkpoint["args"].items():
            if key not in {"resume", "eval", "output", "device", "workers", "data"}:
                if str(getattr(args, key, None)) != str(old):
                    raise ValueError(f"Resume configuration differs: {key}: {old}")
        model.load_state_dict(checkpoint["model"])
        if ema is not None:
            ema.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], train_loader.generator)
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
    write_json(args.output / "config.json", {k: str(v) if hasattr(v, "__fspath__") else v
                                            for k, v in vars(args).items()})
    steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
    for epoch in range(start, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = correct = 0
        loss_sum = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if i >= steps:
                break
            value = scheduled_offset(epoch * steps + i, args.ramp_epochs * steps,
                                     args.final_offset, args.schedule)
            set_offset(model, value)
            if scheduler_each_batch:
                position = epoch * steps + i
                lrs = [base * fn(position) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
                for group, lr in zip(optimizer.param_groups, lrs):
                    group["lr"] = lr
                scheduler.last_epoch, scheduler._last_lr = position, lrs
            reset_net(model)
            x, y = x.to(device), y.to(device).reshape(-1)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=args.smoothing)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {i}.")
            group_start = (i // accumulation) * accumulation
            group_size = min(accumulation, steps - group_start)
            scaler.scale(loss / group_size).backward()
            if (i + 1) % accumulation == 0 or i + 1 == steps:
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    with torch.no_grad():
                        current_state = model.state_dict()
                        for key, value in ema.state_dict().items():
                            current = current_state[key]
                            if value.is_floating_point() and not key.endswith('.offset'):
                                value.mul_(decay).add_(current, alpha=1-decay)
                            else:
                                value.copy_(current)
            reset_net(model)
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
        if not scheduler_each_batch:
            scheduler.step()
        # Evaluate and checkpoint with the exact offset at this epoch boundary.
        offset = scheduled_offset((epoch + 1) * steps, args.ramp_epochs * steps,
                                  args.final_offset, args.schedule)
        set_offset(model, offset)
        metrics = evaluate(model, selection_loader, device, args.max_batches)
        ema_metrics = None
        if ema is not None:
            set_offset(ema, offset)
            ema_metrics = evaluate(ema, selection_loader, device, args.max_batches)
        row = dict(epoch=epoch, offset=offset, lr=optimizer.param_groups[0]["lr"],
                   train_loss=loss_sum / total, train_accuracy=100 * correct / total,
                   selection=metrics, ema_selection=ema_metrics)
        print(json.dumps(row))
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        selected = ema_metrics if ema_metrics and ema_metrics["accuracy"] > metrics["accuracy"] else metrics
        improved = selected["accuracy"] > best and epoch >= args.select_after
        if improved:
            best = selected["accuracy"]
        extra = {"ema": ema.state_dict()} if ema is not None else {}
        save_checkpoint(args.output / "last.pt", model, optimizer, scheduler, scaler,
                        args, epoch, best, train_loader.generator, **extra)
        if improved:
            best_model = ema if ema_metrics and selected is ema_metrics else model
            save_checkpoint(args.output / "best.pt", best_model, optimizer, scheduler, scaler,
                            args, epoch, best, train_loader.generator, **extra)
    best_path = args.output / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
        result = evaluate(model, test_loader, device, args.max_batches)
        write_json(args.output / "test.json", result)
        print("Best checkpoint test:", json.dumps(result))

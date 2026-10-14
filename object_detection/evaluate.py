"""Frame-based YOLO mAP50 and mAP50:95, matching the original experiment metric."""
import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms

from object_detection.boxes import xywh2xyxy
from oilif import SpikeCounter
from object_detection.metrics import ap_per_class, box_iou


def nms(predictions, confidence=0.001, iou=0.7, max_det=300):
    results = []
    for prediction in predictions.transpose(1, 2):
        boxes, scores = xywh2xyxy(prediction[:, :4]), prediction[:, 4:]
        anchor, cls = torch.where(scores > confidence)
        score = scores[anchor, cls]
        if len(score) > 30000:
            order = score.argsort(descending=True)[:30000]
            anchor, cls, score = anchor[order], cls[order], score[order]
        boxes = boxes[anchor].float()
        # Use the original class offset and candidate order, including ties.
        keep = torchvision_nms(boxes + cls[:, None] * 7680, score.float(), iou)[:max_det]
        results.append(torch.cat((boxes[keep], score[keep, None], cls[keep, None]), 1))
    return results


def match(detections, targets, thresholds):
    correct = np.zeros((len(detections), len(thresholds)), dtype=bool)
    if not len(targets) or not len(detections):
        return correct
    iou = box_iou(targets[:, 1:], detections[:, :4]).cpu().numpy()
    same_class = (targets[:, :1] == detections[:, 5]).cpu().numpy()
    for i, threshold in enumerate(thresholds):
        label, det = np.where((iou >= threshold) & same_class)
        pairs = np.stack((label, det, iou[label, det]), 1)
        if len(pairs):
            pairs = pairs[pairs[:, 2].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
            correct[pairs[:, 1].astype(int), i] = True
    return correct


def native_boxes(boxes, geometry):
    """Undo letterboxing and clip to sensor bounds, as in the source validator."""
    h, w, gain, left, top = geometry
    boxes = (boxes - torch.stack((left, top, left, top))) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes


@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=0):
    model.eval()
    stats = []
    activity = SpikeCounter(model)
    thresholds = np.linspace(0.5, 0.95, 10)
    samples = 0
    for step, batch in enumerate(data_loader):
        if max_batches and step >= max_batches:
            break
        images = batch["img"].to(device=device, dtype=next(model.parameters()).dtype)
        predictions, _ = model(images)
        detections = nms(predictions)
        for i, det in enumerate(detections):
            selected = batch["batch_idx"] == i
            scale = torch.tensor([images.shape[-1], images.shape[-2], images.shape[-1], images.shape[-2]], device=device)
            boxes = xywh2xyxy(batch["bboxes"][selected].to(device)) * scale
            if "geometry" in batch:
                geometry = batch["geometry"][i].to(device)
                boxes = native_boxes(boxes, geometry)
                det = det.clone()
                det[:, :4] = native_boxes(det[:, :4], geometry)
            targets = torch.cat((batch["cls"][selected].to(device), boxes), 1)
            correct = match(det, targets, thresholds)
            stats.append((correct, det[:, 4].cpu().numpy(), det[:, 5].cpu().numpy(),
                          targets[:, 0].cpu().numpy()))
            samples += 1
    rates = activity.result(); activity.close()
    if not stats:
        raise ValueError("Empty evaluation split.")
    tp, confidence, cls, target_cls = [np.concatenate(parts) for parts in zip(*stats)]
    if not len(confidence) or not len(target_cls):
        return dict(map50=0.0, map50_95=0.0, samples=samples, **rates)
    ap = ap_per_class(tp, confidence, cls, target_cls)[5]
    return dict(map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()), samples=samples, **rates)

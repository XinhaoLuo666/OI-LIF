# Adapted from Ultralytics 8.0.197 / SpikeYOLO; AGPL-3.0.
# Original file: ultralytics/utils/metrics.py
import math
from pathlib import Path
import numpy as np
import torch

def box_iou(box1, box2, eps=1e-07):
    (a1, a2), (b1, b2) = (box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2))
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (box1.chunk(4, -1), box2.chunk(4, -1))
        w1_, h1_, w2_, h2_ = (w1 / 2, h1 / 2, w2 / 2, h2 / 2)
        b1_x1, b1_x2, b1_y1, b1_y2 = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_)
        b2_x1, b2_x2, b2_y1, b2_y2 = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1, b1_y2 - b1_y1 + eps)
        w2, h2 = (b2_x2 - b2_x1, b2_y2 - b2_y1 + eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = 4 / math.pi ** 2 * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return (ap, mpre, mrec)

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16, prefix=''):
    i = np.argsort(-conf)
    tp, conf, pred_cls = (tp[i], conf[i], pred_cls[i])
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = (np.linspace(0, 1, 1000), [])
    ap, p_curve, r_curve = (np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    prec_values = np.array(prec_values)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = (p_curve[:, i], r_curve[:, i], f1_curve[:, i])
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return (tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values)

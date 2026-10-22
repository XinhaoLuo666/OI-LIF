"""Gen1 .dat / bbox.npy -> four last-event frames and normalized YOLO boxes.

DAT decoding follows Prophesee's Apache-2.0 event format: uint32 timestamp,
uint32 packed x (14 bits), y (14 bits), polarity (1 bit).
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_dat(path):
    with path.open("rb") as stream:
        comments = 0
        while True:
            position = stream.tell()
            if not stream.readline().startswith(b"% "):
                stream.seek(position)
                break
            comments += 1
        if comments:
            _, event_size = stream.read(2)
            if event_size != 8:
                raise ValueError(f"Unsupported DAT event size: {event_size}")
        offset = stream.tell()
    return np.memmap(path, mode="r", offset=offset, dtype=[("t", "<u4"), ("packed", "<u4")])


def render(events, start, stop, height=240, width=304):
    packed = events["packed"][start:stop]
    x = (packed & 0x3FFF).astype(np.int64)
    y = ((packed >> 14) & 0x3FFF).astype(np.int64)
    polarity = ((packed >> 28) & 1).astype(np.uint8)
    valid = (x < width) & (y < height)
    x, y, polarity = x[valid], y[valid], polarity[valid]
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    # Retain the last event per pixel deterministically, including duplicates.
    pixel = y * width + x
    _, reverse_index = np.unique(pixel[::-1], return_index=True)
    last = len(pixel) - 1 - reverse_index
    frame[y[last], x[last]] = 255 * polarity[last, None]
    return frame


def labels_for(boxes, height=240, width=304):
    if "track_id" in boxes.dtype.names:
        _, indices = np.unique(boxes["track_id"][::-1], return_index=True)
        boxes = boxes[np.sort(len(boxes) - 1 - indices)]
    x = np.clip(boxes["x"].astype(float), 0, width)
    y = np.clip(boxes["y"].astype(float), 0, height)
    w = np.clip(boxes["w"].astype(float), 0, width)
    h = np.clip(boxes["h"].astype(float), 0, height)
    # Match the legacy converter's center/size convention after clipping.
    cx, cy = np.clip(x + w / 2, 0, width), np.clip(y + h / 2, 0, height)
    labels = np.stack((boxes["class_id"], cx / width, cy / height, w / width, h / height), 1)
    return labels[(w > 0) & (h > 0)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--window-us", type=int, default=250000)
    args = parser.parse_args()
    if args.window_us <= 0 or args.time_steps <= 0 or args.window_us % args.time_steps:
        raise ValueError("window-us must be positive and divisible by time-steps.")
    for split in args.splits:
        output = args.output / split
        output.mkdir(parents=True, exist_ok=True)
        paths = sorted((args.input / split).rglob("*_bbox.npy"))
        if not paths:
            raise FileNotFoundError(f"No Gen1 annotations in {args.input / split}")
        for path in tqdm(paths, desc=split):
            sequence = path.name.removesuffix("_bbox.npy")
            events = read_dat(path.with_name(sequence + "_td.dat"))
            boxes = np.load(path, allow_pickle=False)
            time_key = "t" if "t" in boxes.dtype.names else "ts"
            timestamps, starts = np.unique(boxes[time_key], return_index=True)
            for index, group in enumerate(np.split(boxes, starts[1:])):
                timestamp = int(timestamps[index])
                boundaries = timestamp - args.window_us + np.arange(args.time_steps + 1) * (args.window_us // args.time_steps)
                positions = np.searchsorted(events["t"], boundaries, side="left")
                frames = np.stack([render(events, positions[t], positions[t + 1]) for t in range(args.time_steps)])
                # Preserve the existing cached data layout [T,W,H,C].
                frames = frames.transpose(0, 2, 1, 3)
                target = output / f"{sequence}_{index:06d}"
                if target.with_suffix(".npy").exists():
                    raise FileExistsError(f"Refusing to overwrite {target}")
                np.save(target.with_suffix(".npy"), frames, allow_pickle=False)
                np.savetxt(target.with_suffix(".txt"), labels_for(group), fmt="%.8g")


if __name__ == "__main__":
    main()

"""Box precision, recall and F1 (Section IV-B of the paper).

    python evaluate.py BOXES LABELS [--end-ms MS]          # DVXplorer clips
    python evaluate.py BOXES LABELS --ictrap SCENE.h5      # ICTrap
"""

import argparse
import collections
import csv

import numpy as np

STEP_US = 10_000  # one frame of our labels and one step of the tracker


class Box(tuple):
    """(x1, y1, x2, y2) with a `certain` flag."""
    certain = True


def make_box(x1, y1, x2, y2, certain=True):
    box = Box((x1, y1, x2, y2))
    box.certain = certain
    return box


def load_labels(path):
    """{frame_index: [Box, ...]} from a label CSV."""
    labels = collections.defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            x, y = float(row['left']), float(row['top'])
            labels[int(row['frame_index'])].append(make_box(
                x, y, x + float(row['width']), y + float(row['height']),
                row.get('confidence', 'certain') == 'certain'))
    return labels


def load_predictions(path):
    """{timestamp_us: [Box, ...]} from the tracker's box CSV."""
    boxes = collections.defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            boxes[int(row['timestamp'])].append(
                make_box(*(int(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))))
    return boxes


def centre_in_circle(label, pred):
    """Distance between the centres, and whether it is within the enclosing circle."""
    lx1, ly1, lx2, ly2 = label
    px1, py1, px2, py2 = pred
    d = np.hypot((px1 + px2) / 2 - (lx1 + lx2) / 2, (py1 + py2) / 2 - (ly1 + ly2) / 2)
    return d, d <= np.hypot(lx2 - lx1, ly2 - ly1) / 2


def score_frame(labels, preds, protocol):
    """(TP, FP, FN) of one frame."""
    match_uncertain = protocol == 'dvxplorer'
    pairs = []
    for li, label in enumerate(labels):
        if not (label.certain or match_uncertain):
            continue
        for pi, pred in enumerate(preds):
            d, inside = centre_in_circle(label, pred)
            if inside:
                pairs.append((d, li, pi))
    pairs.sort()
    used_l, used_p = set(), set()
    for _d, li, pi in pairs:
        if li not in used_l and pi not in used_p:
            used_l.add(li)
            used_p.add(pi)

    tp = len(used_l)
    fn = sum(1 for li, label in enumerate(labels) if label.certain and li not in used_l)
    if match_uncertain:
        fp = len(preds) - tp
    else:
        regions = [label for label in labels if not label.certain]
        fp = sum(1 for pi, pred in enumerate(preds) if pi not in used_p
                 and not any(centre_in_circle(r, pred)[1] for r in regions))
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def score_frames(frames, protocol):
    """Sums score_frame over (labels, preds) pairs."""
    tp = fp = fn = 0
    for labels, preds in frames:
        a, b, c = score_frame(labels, preds, protocol)
        tp, fp, fn = tp + a, fp + b, fn + c
    p, r, f1 = prf(tp, fp, fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}


def evaluate_dvxplorer(boxes_path, labels_path, end_us, step_us=STEP_US):
    """DVXplorer clips: every 10 ms step up to end_us is a frame, empty ones included."""
    labels = {(n + 1) * step_us: b for n, b in load_labels(labels_path).items()}
    preds = load_predictions(boxes_path)
    last = max([end_us] + list(labels))
    steps = sorted(set(range(step_us, last + step_us, step_us)) | set(labels))
    return score_frames(((labels.get(ts, []), preds.get(ts, [])) for ts in steps),
                        'dvxplorer')


def load_triggers(h5_path):
    """{RGB frame index: start time}, zero-based on the first event."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        t0 = int(f['CD/events'][0]['t'])
        triggers = f['EXTERNAL_TRIGGERS/corrected_positive'][:]
    return {int(row['f']): int(row['t']) - t0 for row in triggers}


def evaluate_ictrap(boxes_path, annotation_path, h5_path, step_us=STEP_US):
    """ICTrap: each RGB frame is scored with the boxes of the first step after it ends."""
    triggers = load_triggers(h5_path)
    labels = load_labels(annotation_path)
    preds = load_predictions(boxes_path)
    frames = [n for n in sorted(triggers) if n >= 0 and n + 1 in triggers]
    step_of = {n: -(-triggers[n + 1] // step_us) * step_us for n in frames}
    return score_frames(((labels.get(n, []), preds.get(step_of[n], [])) for n in frames),
                        'ictrap')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('boxes', help="the tracker's box CSV")
    parser.add_argument('labels', help='label CSV')
    parser.add_argument('--end-ms', type=float, default=None,
                        help='length of the recording; steps up to here are scored, empty '
                             'ones included (default: the last labeled or predicted step)')
    parser.add_argument('--ictrap', metavar='SCENE.h5', default=None,
                        help='score an ICTrap scene; its .h5 file gives the frame times')
    args = parser.parse_args()

    if args.ictrap:
        m = evaluate_ictrap(args.boxes, args.labels, args.ictrap)
    else:
        if args.end_ms is not None:
            end_us = int(round(args.end_ms * 1000))
        else:
            labeled = [(n + 1) * STEP_US for n in load_labels(args.labels)]
            end_us = max(labeled + list(load_predictions(args.boxes)) + [0])
        m = evaluate_dvxplorer(args.boxes, args.labels, end_us)
    print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  F1={m['f1']:.3f}")


if __name__ == '__main__':
    main()

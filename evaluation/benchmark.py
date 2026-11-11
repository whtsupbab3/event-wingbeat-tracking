"""Runs the tracker on the 21 labeled clips and writes our results of Table I.

    make -C ../tracker
    python benchmark.py --data DATA --ictrap ICTRAP [--out results.md]
    python benchmark.py ... --max_k 1               # ablation, Section V-C
    python benchmark.py ... --no-background_gate    # ablation, Section V-C

DATA holds event_streams/<clip>.aedat4 and labels/<clip>.csv; ICTRAP holds
<scene>.h5 and <scene>_annotation.csv. Unknown flags go to tracker/track.
Speed is the best of --repeats runs, processing only.
"""

import argparse
import collections
import json
import os
import sys
import tempfile

import numpy as np

import evaluate
import events
import run

Clip = collections.namedtuple('Clip', 'name group stem')

# Table I, in order. stem is the file name of the clip.
CLIPS = [
    Clip('Bee 1', 'Single bee', 'bee1'),  # a hand moves through the view
    Clip('Bee 2', 'Single bee', 'bee2_1min'),
    Clip('Bee 3', 'Single bee', 'bee3_0'),
    Clip('Bee 4', 'Single bee', 'bee3_3'),
    Clip('Bee 5', 'Single bee', 'bee3_4'),
    Clip('Bee 6', 'Single bee', 'bee4_1'),
    Clip('Bee 7', 'Single bee', 'bee4_3'),
    Clip('Bee 8', 'Single bee', 'bee4_4'),
    Clip('Bee 9', 'Single bee', 'bee4_5'),
    Clip('Synthetic 1', 'Synthetic multi-bee', 'multi_bee'),
    Clip('Synthetic 2', 'Synthetic multi-bee', 'multi_bee2'),
    Clip('Synthetic 3', 'Synthetic multi-bee', 'multi_bee3'),
    Clip('Synthetic 4', 'Synthetic multi-bee', 'multi_bee4'),
    Clip('Garden', 'Real multi-insect', 'backyard_bees'),
    Clip('Hive entrance', 'Real multi-insect', 'horiuchi_event_log_10_12_freq_250ms_fixed_60s'),
    Clip('1_l-l-l', 'ICTrap', '1_l-l-l'),
    Clip('2_l-h-l', 'ICTrap', '2_l-h-l'),
    Clip('3_m-h-h', 'ICTrap', '3_m-h-h'),
    Clip('4_m-m-h', 'ICTrap', '4_m-m-h'),
    Clip('5_h-l-h', 'ICTrap', '5_h-l-h'),
    Clip('6_h-h-h', 'ICTrap', '6_h-h-h'),
]
GROUPS = ['Single bee', 'Synthetic multi-bee', 'Real multi-insect', 'ICTrap']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='folder with event_streams/ and labels/')
    parser.add_argument('--ictrap', required=True, help='folder with the ICTrap scenes')
    parser.add_argument('--out', default='results.md', help='Markdown table to write')
    parser.add_argument('--boxes-dir', default=None,
                        help='keep the box CSVs here (default: a temporary folder)')
    parser.add_argument('--clips', nargs='+', default=None,
                        help='only these clips (file stems, e.g. bee1 1_l-l-l)')
    parser.add_argument('--repeats', type=int, default=3, help='runs per clip for the speed')
    parser.add_argument('--config', default=run.CONFIG, help='parameter file for tracker/track')
    return parser.parse_known_args()


def paths(clip, args):
    """(events, labels, width, height)."""
    if clip.group == 'ICTrap':
        base = os.path.join(args.ictrap, clip.stem)
        return base + '.h5', base + '_annotation.csv', 1280, 720
    return (os.path.join(args.data, 'event_streams', clip.stem + '.aedat4'),
            os.path.join(args.data, 'labels', clip.stem + '.csv'), 640, 480)


def run_clip(clip, args, extra_args, boxes_dir, tmp):
    events_path, labels, width, height = paths(clip, args)
    ev = events.load(events_path)
    duration_us = int(ev.t[-1] - ev.t[0])
    events_bin = os.path.join(tmp, 'events.bin')
    events.write_track_input(ev, width, height, events_bin)
    del ev

    boxes = os.path.join(boxes_dir, clip.stem + '.csv')
    busy = min(run.run_track(events_bin, boxes, extra_args, args.config)
               for _ in range(args.repeats))
    if clip.group == 'ICTrap':
        m = evaluate.evaluate_ictrap(boxes, labels, events_path)
    else:
        m = evaluate.evaluate_dvxplorer(boxes, labels, duration_us)
    m.update(duration_s=duration_us * 1e-6, busy_s=busy)
    return m


def f1_cell(f1, precision, recall):
    return f'{f1 * 100:.1f} ({precision * 100:.1f} / {recall * 100:.1f})'


def write_table(path, results, extra_args):
    lines = ['# Results per clip (Table I)', '',
             f"tracker/track args: `{' '.join(extra_args) or 'config only'}`. "
             'Box F1 (precision / recall) in %; speed = recording length / processing time.', '',
             '| Clip | Group | Box F1 (P / R) | Speed |', '|---|---|---|---|']
    for clip in CLIPS:
        if clip.stem in results:
            m = results[clip.stem]
            lines.append(f"| {clip.name} | {clip.group} | "
                         f"{f1_cell(m['f1'], m['precision'], m['recall'])} | "
                         f"{m['duration_s'] / m['busy_s']:.0f} |")

    lines += ['', '## Means', '',
              'F1, precision and recall are means over the clips; speed is total length / '
              'total processing time.', '',
              '| Clips | n | Box F1 (P / R) | Speed |', '|---|---|---|---|']
    sets = [(g, [results[c.stem] for c in CLIPS if c.group == g and c.stem in results])
            for g in GROUPS]
    sets.append(('All', [results[c.stem] for c in CLIPS if c.stem in results]))
    for name, rows in sets:
        if rows:
            means = [float(np.mean([r[k] for r in rows])) for k in ('f1', 'precision', 'recall')]
            speed = sum(r['duration_s'] for r in rows) / sum(r['busy_s'] for r in rows)
            lines.append(f'| {name} | {len(rows)} | {f1_cell(*means)} | {speed:.0f} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args, extra_args = parse_args()
    clips = [c for c in CLIPS if not args.clips or c.stem in args.clips]
    if args.clips and len(clips) != len(args.clips):
        known = {c.stem for c in CLIPS}
        raise SystemExit(f'unknown clips: {sorted(set(args.clips) - known)}')

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        boxes_dir = args.boxes_dir or tmp
        os.makedirs(boxes_dir, exist_ok=True)
        for clip in clips:
            m = run_clip(clip, args, extra_args, boxes_dir, tmp)
            results[clip.stem] = m
            print(f"{clip.name:14s} {clip.group:20s} F1 {m['f1'] * 100:5.1f}  "
                  f"({m['duration_s'] / m['busy_s']:.0f}x)", flush=True)

    write_table(args.out, results, extra_args)
    json_path = os.path.splitext(args.out)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump({'track_args': extra_args, 'clips': results}, f, indent=1)
    print(f'wrote {args.out} and {json_path}', file=sys.stderr)


if __name__ == '__main__':
    main()

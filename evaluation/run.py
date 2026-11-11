"""Runs the C tracker on one event file and, given labels, scores its boxes.

    make -C ../tracker
    python run.py EVENTS [--labels LABELS.csv] [--boxes BOXES.csv] [--<param> VALUE ...]

Unknown flags (--sigma_p 600, --no-background_gate, ...) go to tracker/track
and override configs/default.conf. A .h5 file is scored as an ICTrap scene.
"""

import argparse
import os
import re
import subprocess
import tempfile

import evaluate
import events

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRACK = os.path.join(ROOT, 'tracker', 'track')
CONFIG = os.path.join(ROOT, 'configs', 'default.conf')


def run_track(events_bin, boxes_csv, extra_args=(), config=CONFIG):
    """Runs tracker/track; returns its processing time in seconds."""
    if not os.path.exists(TRACK):
        raise SystemExit(f'{TRACK} not found; build it with `make -C {os.path.dirname(TRACK)}`')
    r = subprocess.run([TRACK, '--config', config, '--bbox_csv', boxes_csv, *extra_args,
                        events_bin], check=True, capture_output=True, text=True)
    return float(re.search(r'busy_s=([\d.]+)', r.stdout).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='event file (.aedat4, .raw, ICTrap .h5, ...)')
    parser.add_argument('--width', type=int, default=None,
                        help='sensor width (default: 1280 for .h5, else 640)')
    parser.add_argument('--height', type=int, default=None,
                        help='sensor height (default: 720 for .h5, else 480)')
    parser.add_argument('--labels', default=None, help='label CSV to score the boxes against')
    parser.add_argument('--boxes', default=None,
                        help='write the boxes here (default: a temporary file)')
    parser.add_argument('--config', default=CONFIG, help='parameter file for tracker/track')
    args, extra_args = parser.parse_known_args()

    ictrap = args.source.endswith('.h5')
    width = args.width or (1280 if ictrap else 640)
    height = args.height or (720 if ictrap else 480)

    ev = events.load(args.source)
    duration_us = int(ev.t[-1] - ev.t[0])
    with tempfile.TemporaryDirectory() as tmp:
        events_bin = os.path.join(tmp, 'events.bin')
        events.write_track_input(ev, width, height, events_bin)
        del ev
        boxes = args.boxes or os.path.join(tmp, 'boxes.csv')
        busy = run_track(events_bin, boxes, extra_args, args.config)
        print(f'{duration_us * 1e-6:.1f} s of events in {busy:.3f} s: '
              f'{duration_us * 1e-6 / busy:.0f}x real time')
        if args.labels:
            if ictrap:
                m = evaluate.evaluate_ictrap(boxes, args.labels, args.source)
            else:
                m = evaluate.evaluate_dvxplorer(boxes, args.labels, duration_us)
            print(f"box F1 {m['f1'] * 100:.1f}  (P {m['precision'] * 100:.1f} / "
                  f"R {m['recall'] * 100:.1f}; TP={m['tp']} FP={m['fp']} FN={m['fn']})")


if __name__ == '__main__':
    main()

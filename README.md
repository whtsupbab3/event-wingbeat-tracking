# Neuromorphic Detection of Flying Insects by Wingbeat Frequency

Code for the paper *Neuromorphic Detection of Flying Insects by Wingbeat
Frequency*.

The tracker finds flying insects in event-camera recordings by the beat of
their wings, not by how they move. Each pixel measures how often its
brightness changes, and only pixels that flicker at about the wingbeat
frequency are passed on to the trackers. A second check hides boxes that have
many events but no wingbeat, such as a branch moving in the wind.

| Folder | What is in it |
|---|---|
| [`tracker/`](tracker) | The tracker, in plain C with no extra libraries |
| [`configs/default.conf`](configs/default.conf) | All parameters, named as in the paper. We used the same values for every result. |
| [`evaluation/`](evaluation) | Reading event files, scoring, and the script that produces Table I |

## Paper sections in the code

| Paper | Code |
|---|---|
| III-A Period at each pixel | `tracking_update()` in [`tracker/tracking.c`](tracker/tracking.c) |
| III-A Hot pixels | `drop_hot_pixels()` in [`evaluation/events.py`](evaluation/events.py) |
| III-B Weight and position update (Eq. 1–4) | `step()` and `accept()` in `tracking.c` |
| III-C Boxes | `accept()` and `get_bbox()` in `tracking.c` |
| III-D Several trackers | `step()` and `create_tracker()` in `tracking.c` |
| III-E Background test (Eq. 5–6) | `is_background()` and `spectral_peak()` in `tracking.c` |
| IV-B Scoring | [`evaluation/evaluate.py`](evaluation/evaluate.py) |
| IV-D, V-A, V-C Speed, Table I, ablation | [`evaluation/benchmark.py`](evaluation/benchmark.py) |

## Running the tracker

```sh
make -C tracker
pip install -r evaluation/requirements.txt
python evaluation/run.py recording.aedat4 --labels labels.csv --boxes boxes.csv
```

`run.py` removes hot pixels, runs the tracker and, if you give it labels,
scores the result. You can change any parameter on the command line, for
example `--sigma_p 600` or `--no-background_gate`. The output has one box per
tracked insect every 10 ms, as `timestamp,track_id,x1,y1,x2,y2`.

## Reproducing the results

```sh
python evaluation/benchmark.py --data DATA --ictrap ICTRAP --out results.md
```

`DATA` contains the DVXplorer clips (`event_streams/<clip>.aedat4` and
`labels/<clip>.csv`). `ICTRAP` contains the six scenes of the
[ICTrap dataset](https://www.doi.org/10.17879/07938503301) by Gebauer et al.
(`<scene>.h5` and `<scene>_annotation.csv`). For scene 6, use the public,
privacy-filtered version.

For the ablations in Section V-C, add one of `--max_k 1`, `--max_k 2` or
`--no-background_gate`.

F1, precision and recall are the same on every run. Speed depends on the
computer; the paper used one core of an Apple M4.

### How boxes are scored

A predicted box counts as correct if its centre lies inside the smallest
circle around a labeled box. Each labeled box can be matched only once, and
the closest pairs are matched first. Boxes labeled `uncertain` (insects that
sit or crawl) are treated differently in the two datasets:

- **DVXplorer clips**: an uncertain box counts as correct if it is found and is
  ignored if it is missed. Every 10 ms step is scored.
- **ICTrap**: we follow Gebauer et al. Uncertain boxes are never counted, and
  a predicted box on one is not an error. Each camera frame is scored with the
  boxes from the first 10 ms step after the frame ends.

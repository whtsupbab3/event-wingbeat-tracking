"""Reads event files (ICTrap .h5, or anything `faery` reads), removes hot
pixels (Section III-A) and writes the input of tracker/track."""

import collections
import struct

import numpy as np

Events = collections.namedtuple('Events', ['t', 'x', 'y', 'p'])

# Hot pixels fire in at least this fraction of the 1 s intervals.
HOT_PIXEL_FRAC = 0.99


def read(path):
    """Events of a file, t in microseconds from the first event."""
    if path.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            ev = f['CD/events'][:]
        t, x, y, p = ev['t'], ev['x'], ev['y'], ev['p']
    else:
        import faery
        ev = faery.events_stream_from_file(path).to_array()
        t, x, y, p = ev['t'], ev['x'], ev['y'], ev['p']
    return Events((t - t[0]).astype(np.int32), x.astype(np.uint16), y.astype(np.uint16),
                  p.astype(bool))


def drop_hot_pixels(ev, min_active=HOT_PIXEL_FRAC, bin_us=1_000_000):
    """Drops the events of pixels that fire in >= min_active of all 1 s bins."""
    t, x, y = ev.t.astype(np.int64), ev.x.astype(np.int64), ev.y.astype(np.int64)
    width = int(x.max()) + 1
    n_px = width * (int(y.max()) + 1)
    idx = y * width + x
    b = (t - t[0]) // bin_us
    n_bins = int(b.max()) + 1
    active = np.bincount(np.unique(idx * n_bins + b) // n_bins, minlength=n_px)
    hot = active >= min_active * n_bins
    keep = ~hot[idx]
    return Events(*(a[keep] for a in ev))


def load(path):
    """read() and drop_hot_pixels(), as used for every clip in the paper."""
    return drop_hot_pixels(read(path))


def write_track_input(ev, width, height, path):
    """Writes the binary event file that tracker/track reads."""
    if np.any(np.diff(ev.t) < 0):
        raise ValueError('event times are not sorted')
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sIIIQ', b'IWEV', width, height, 0, len(ev.t)))
        for a, dtype in zip(ev, ('<i4', '<u2', '<u2', 'u1')):
            np.ascontiguousarray(a, dtype).tofile(f)

// Wingbeat tracker (Section III of the paper).
#ifndef TRACKING_H
#define TRACKING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Parameters, described in configs/default.conf.
typedef struct {
	int width;
	int height;
	double target_freq;
	double sigma_d;
	double sigma_p;
	double wmax;
	double lock_bonus;
	double spawn_dist;
	double spawn_max_dp;
	int64_t unlocked_grace_us;
	double bbox_interval_ms;
	int64_t max_period_us;

	int64_t bbox_window_us;
	double bbox_radius;
	int64_t bbox_min_matches;
	int64_t bbox_pad;
	double wmin;
	int64_t max_k;

	// Background test (Section III-E)
	bool background_gate;
	int64_t bg_window_us;
	int64_t bg_bin_us;
	int64_t bg_pad;
	int64_t bg_min_events;
	double bg_fmin;
	double bg_fmax;
	double bg_max_peak;
	double bg_clear_peak;
	double bg_small_frac;
	double bg_inherit_px;
	int64_t bg_inherit_us;
} TrackingParams;

typedef struct {
	int64_t ts;
	int id;
	int x1, y1, x2, y2;
} BoxRow;

// Growing list of output rows; the caller owns it.
typedef struct {
	BoxRow *rows;
	size_t len, cap;
} BoxList;

typedef struct Tracking Tracking;

void tracking_default_params(TrackingParams *p, int width, int height);
Tracking *tracking_new(const TrackingParams *p);
void tracking_free(Tracking *trk);

// One chunk of events, sorted by time. Appends the boxes that are now due.
void tracking_update(Tracking *trk, const int32_t *t, const uint16_t *x,
                     const uint16_t *y, const uint8_t *p, size_t n, BoxList *out);
// Appends the rows due up to end_ts, for the end of a file.
void tracking_flush(Tracking *trk, int64_t end_ts, BoxList *out);
int tracking_num_trackers(const Tracking *trk);

#endif

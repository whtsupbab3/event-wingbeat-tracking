#include "tracking.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int32_t t;
	int16_t x, y;
} Match;

// Period map of one pixel (Section III-A), kept in one cache line.
typedef struct {
	int32_t on2off, off2on;
	int32_t period, ts;
} Pixel;

typedef struct {
	int32_t t;
	uint16_t x, y;
} Event;

// Recent events in 16x16 px cells, in time order, for the background test.
#define CELL_SHIFT 4
typedef struct {
	Event *ev;
	size_t len, cap;
} Cell;

// Events [i0, i1) of a cell that fall in the gate's window.
typedef struct {
	const Cell *c;
	size_t i0, i1;
} Span;

typedef struct {
	int64_t ts;
	double cx, cy;
} HiddenBox;

typedef struct {
	int id;
	double x_hat, y_hat;
	int64_t last_locked_ts;
	bool locked;
	// -1 until the gate first looks at it, then 1 if the box is hidden
	int background;
	// Recent matches, a ring buffer with a power-of-two size.
	Match *m;
	size_t head, len, cap;
} Tracker;

struct Tracking {
	TrackingParams p;
	int64_t interval_us;
	double p_ideal, inv_sd2, inv_sp2, spawn_r2, inherit_r2;

	Pixel *px;

	bool started;
	int64_t last_seen, next_bbox_ts;
	int next_id;

	Tracker *tr;
	size_t n_tr, cap_tr;

	// Measurement scratch: pixel, timestamp, sort order.
	uint32_t *m_pix;
	int32_t *m_ts;
	intptr_t *m_order;
	size_t m_cap;

	// Background test. Events before keep_from are never needed again.
	Cell *cells;
	Span *spans;
	int n_cx, n_cy;
	int64_t keep_from;
	bool ev_alive;
	HiddenBox *hidden;
	size_t hid_start, hid_end, hid_cap;

	// Spectrum of the event rate.
	int n_bins, k0, k1;
	int32_t *bins;
	double *window, *counts, *re, *im, *cos_t, *sin_t;
};

static void *xrealloc(void *ptr, size_t size)
{
	ptr = realloc(ptr, size);
	if (ptr == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(1);
	}
	return ptr;
}

static size_t grown(size_t cap, size_t need)
{
	cap = cap ? cap : 64;
	while (cap < need)
		cap *= 2;
	return cap;
}

void tracking_default_params(TrackingParams *p, int width, int height)
{
	*p = (TrackingParams){
		.width = width,
		.height = height,
		.target_freq = 220.0,
		.sigma_d = 80.0,
		.sigma_p = 900.0,
		.wmax = 0.3,
		.lock_bonus = 3.0,
		.spawn_dist = 60.0,
		.spawn_max_dp = 300.0,
		.unlocked_grace_us = 100000,
		.bbox_interval_ms = 10.0,
		.max_period_us = 32768,
		.bbox_window_us = 20000,
		.bbox_radius = 17,
		.bbox_min_matches = 5,
		.bbox_pad = 3,
		.wmin = 1e-6,
		.max_k = 3,
		.background_gate = true,
		.bg_window_us = 128000,
		.bg_bin_us = 500,
		.bg_pad = 4,
		.bg_min_events = 8000,
		.bg_fmin = 50.0,
		.bg_fmax = 850.0,
		.bg_max_peak = 6.0,
		.bg_clear_peak = 50.0,
		.bg_small_frac = 0.15,
		.bg_inherit_px = 120.0,
		.bg_inherit_us = 300000,
	};
}

Tracking *tracking_new(const TrackingParams *p)
{
	Tracking *trk = calloc(1, sizeof *trk);
	trk->p = *p;
	trk->interval_us = (int64_t)nearbyint(p->bbox_interval_ms * 1000);
	trk->p_ideal = 1e6 / p->target_freq;
	trk->inv_sd2 = 1.0 / (p->sigma_d * p->sigma_d);
	trk->inv_sp2 = 1.0 / (p->sigma_p * p->sigma_p);
	trk->spawn_r2 = p->spawn_dist * p->spawn_dist;
	trk->inherit_r2 = p->bg_inherit_px * p->bg_inherit_px;
	trk->next_bbox_ts = trk->interval_us;
	trk->last_seen = INT32_MIN;

	size_t n_px = (size_t)p->width * p->height;
	trk->px = malloc(n_px * sizeof(Pixel));
	for (size_t i = 0; i < n_px; i++)
		trk->px[i] = (Pixel){INT32_MIN, INT32_MIN, INT32_MAX, INT32_MIN};
	trk->n_cx = ((p->width - 1) >> CELL_SHIFT) + 1;
	trk->n_cy = ((p->height - 1) >> CELL_SHIFT) + 1;
	trk->cells = calloc((size_t)trk->n_cx * trk->n_cy, sizeof(Cell));
	trk->spans = malloc((size_t)trk->n_cx * trk->n_cy * sizeof(Span));

	// Hann window and the bins f_min <= f_n <= f_max of Eq. 6.
	int n = (int)(p->bg_window_us / p->bg_bin_us);
	trk->n_bins = n;
	trk->bins = malloc(n * sizeof(int32_t));
	trk->window = malloc(n * sizeof(double));
	trk->counts = malloc(n * sizeof(double));
	trk->re = malloc(n * sizeof(double));
	trk->im = malloc(n * sizeof(double));
	trk->cos_t = malloc(n * sizeof(double));
	trk->sin_t = malloc(n * sizeof(double));
	for (int k = 0; k < n; k++) {
		trk->window[k] = n == 1 ? 1.0 : 0.5 + 0.5 * cos(M_PI * (1 - n + 2 * k) / (n - 1));
		trk->cos_t[k] = cos(2 * M_PI * k / n);
		trk->sin_t[k] = sin(2 * M_PI * k / n);
	}
	double df = 1.0 / (n * (p->bg_bin_us * 1e-6));
	trk->k0 = n / 2 + 1;
	trk->k1 = -1;
	for (int k = n / 2; k >= 0; k--) {
		if (k * df >= p->bg_fmin && k * df <= p->bg_fmax) {
			trk->k0 = k;
			if (trk->k1 < 0)
				trk->k1 = k;
		}
	}
	return trk;
}

void tracking_free(Tracking *trk)
{
	for (size_t i = 0; i < trk->n_tr; i++)
		free(trk->tr[i].m);
	free(trk->tr);
	free(trk->px);
	free(trk->m_pix);
	free(trk->m_ts);
	free(trk->m_order);
	for (int i = 0; i < trk->n_cx * trk->n_cy; i++)
		free(trk->cells[i].ev);
	free(trk->cells);
	free(trk->spans);
	free(trk->hidden);
	free(trk->bins);
	free(trk->window);
	free(trk->counts);
	free(trk->re);
	free(trk->im);
	free(trk->cos_t);
	free(trk->sin_t);
	free(trk);
}

int tracking_num_trackers(const Tracking *trk)
{
	return (int)trk->n_tr;
}

/*************************************************************
 * Sorting
 */

// numpy's argsort, copied so that readings with equal timestamps are handled in
// the same order as in our Python prototype.
static void argsort_heap(const int32_t *v, intptr_t *tosort, intptr_t n)
{
	intptr_t *a = tosort - 1, i, j, l, tmp;
	for (l = n >> 1; l > 0; --l) {
		tmp = a[l];
		for (i = l, j = l << 1; j <= n;) {
			if (j < n && v[a[j]] < v[a[j + 1]])
				j += 1;
			if (v[tmp] < v[a[j]]) {
				a[i] = a[j];
				i = j;
				j += j;
			} else {
				break;
			}
		}
		a[i] = tmp;
	}
	for (; n > 1;) {
		tmp = a[n];
		a[n] = a[1];
		n -= 1;
		for (i = 1, j = 2; j <= n;) {
			if (j < n && v[a[j]] < v[a[j + 1]])
				j++;
			if (v[tmp] < v[a[j]]) {
				a[i] = a[j];
				i = j;
				j += j;
			} else {
				break;
			}
		}
		a[i] = tmp;
	}
}

#define SWAP(a, b) do { intptr_t tmp_ = (a); (a) = (b); (b) = tmp_; } while (0)

static void argsort(const int32_t *v, intptr_t *tosort, intptr_t num)
{
	intptr_t *pl = tosort, *pr = tosort + num - 1;
	intptr_t *stack[128], **sptr = stack;
	intptr_t *pm, *pi, *pj, *pk, vi;
	int depth[128], *psdepth = depth;
	int cdepth = 0;
	int32_t vp;
	for (intptr_t u = num; u >>= 1;)
		cdepth++;
	cdepth *= 2;

	for (;;) {
		if (cdepth < 0) {
			argsort_heap(v, pl, pr - pl + 1);
			goto stack_pop;
		}
		while ((pr - pl) > 15) {
			pm = pl + ((pr - pl) >> 1);
			if (v[*pm] < v[*pl])
				SWAP(*pm, *pl);
			if (v[*pr] < v[*pm])
				SWAP(*pr, *pm);
			if (v[*pm] < v[*pl])
				SWAP(*pm, *pl);
			vp = v[*pm];
			pi = pl;
			pj = pr - 1;
			SWAP(*pm, *pj);
			for (;;) {
				do
					++pi;
				while (v[*pi] < vp);
				do
					--pj;
				while (vp < v[*pj]);
				if (pi >= pj)
					break;
				SWAP(*pi, *pj);
			}
			pk = pr - 1;
			SWAP(*pi, *pk);
			// Push the larger part, loop on the smaller one.
			if (pi - pl < pr - pi) {
				*sptr++ = pi + 1;
				*sptr++ = pr;
				pr = pi - 1;
			} else {
				*sptr++ = pl;
				*sptr++ = pi - 1;
				pl = pi + 1;
			}
			*psdepth++ = --cdepth;
		}

		for (pi = pl + 1; pi <= pr; ++pi) {
			vi = *pi;
			vp = v[vi];
			pj = pi;
			pk = pi - 1;
			while (pj > pl && vp < v[*pk])
				*pj-- = *pk--;
			*pj = vi;
		}
	stack_pop:
		if (sptr == stack)
			break;
		pr = *(--sptr);
		pl = *(--sptr);
		cdepth = *(--psdepth);
	}
}

// Plain ascending sort of pixel indices.
static void sort_pixels(uint32_t *a, intptr_t n)
{
	while (n > 16) {
		uint32_t lo = a[0], mid = a[n / 2], hi = a[n - 1];
		uint32_t pivot = lo < mid ? (mid < hi ? mid : (lo < hi ? hi : lo))
		                          : (lo < hi ? lo : (mid < hi ? hi : mid));
		intptr_t i = 0, j = n - 1;
		for (;;) {
			while (a[i] < pivot)
				i++;
			while (pivot < a[j])
				j--;
			if (i >= j)
				break;
			uint32_t tmp = a[i];
			a[i++] = a[j];
			a[j--] = tmp;
		}
		// Recurse on the smaller side.
		if (j + 1 < n - j - 1) {
			sort_pixels(a, j + 1);
			a += j + 1;
			n -= j + 1;
		} else {
			sort_pixels(a + j + 1, n - j - 1);
			n = j + 1;
		}
	}
	for (intptr_t i = 1; i < n; i++) {
		uint32_t v = a[i];
		intptr_t j = i;
		for (; j > 0 && v < a[j - 1]; j--)
			a[j] = a[j - 1];
		a[j] = v;
	}
}

/*************************************************************
 * Trackers
 */

static void push_match(Tracker *tr, int32_t t, int x, int y)
{
	if (tr->len == tr->cap) {
		size_t cap = tr->cap * 2;
		Match *m = xrealloc(NULL, cap * sizeof(Match));
		for (size_t i = 0; i < tr->len; i++)
			m[i] = tr->m[(tr->head + i) & (tr->cap - 1)];
		free(tr->m);
		tr->m = m;
		tr->head = 0;
		tr->cap = cap;
	}
	tr->m[(tr->head + tr->len) & (tr->cap - 1)] = (Match){t, (int16_t)x, (int16_t)y};
	tr->len++;
}

// Eq. 4, and a match if the reading is within r_b (Section III-C).
static void accept(const Tracking *trk, Tracker *tr, int32_t t, int x, int y, double w)
{
	tr->x_hat += w * (x - tr->x_hat);
	tr->y_hat += w * (y - tr->y_hat);
	if (hypot(x - tr->x_hat, y - tr->y_hat) < trk->p.bbox_radius) {
		push_match(tr, t, x, y);
		tr->locked = (int64_t)tr->len >= trk->p.bbox_min_matches;
	}
}

static void create_tracker(Tracking *trk, int x, int y, int32_t t)
{
	if (trk->n_tr == trk->cap_tr) {
		trk->cap_tr = grown(trk->cap_tr, trk->n_tr + 1);
		trk->tr = xrealloc(trk->tr, trk->cap_tr * sizeof(Tracker));
	}
	Tracker *tr = &trk->tr[trk->n_tr++];
	*tr = (Tracker){.id = trk->next_id++, .x_hat = x, .y_hat = y,
	                .last_locked_ts = t, .background = -1, .cap = 16};
	tr->m = xrealloc(NULL, tr->cap * sizeof(Match));
	accept(trk, tr, t, x, y, 0.0);
}

// Section III-C: box from the matches of the last T_b; false if fewer than m.
static bool get_bbox(const Tracking *trk, const Tracker *tr, int64_t as_of, int box[4])
{
	const TrackingParams *P = &trk->p;
	int64_t lo = as_of - P->bbox_window_us, count = 0;
	int x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
	for (size_t i = 0; i < tr->len; i++) {
		const Match *m = &tr->m[(tr->head + i) & (tr->cap - 1)];
		if (m->t < lo || m->t > as_of)
			continue;
		count++;
		x0 = m->x < x0 ? m->x : x0;
		x1 = m->x > x1 ? m->x : x1;
		y0 = m->y < y0 ? m->y : y0;
		y1 = m->y > y1 ? m->y : y1;
	}
	if (count < P->bbox_min_matches)
		return false;
	const int pad = (int)P->bbox_pad;
	box[0] = x0 - pad > 0 ? x0 - pad : 0;
	box[1] = y0 - pad > 0 ? y0 - pad : 0;
	box[2] = x1 + pad < P->width - 1 ? x1 + pad : P->width - 1;
	box[3] = y1 + pad < P->height - 1 ? y1 + pad : P->height - 1;
	return true;
}

/*************************************************************
 * Background test (Section III-E)
 */

// In-place radix-2 FFT; n is a power of two.
static void fft(double *re, double *im, int n, const double *c, const double *s)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;
		if (i < j) {
			double t = re[i];
			re[i] = re[j];
			re[j] = t;
			t = im[i];
			im[i] = im[j];
			im[j] = t;
		}
	}
	for (int len = 2; len <= n; len <<= 1) {
		int step = n / len;
		for (int i = 0; i < n; i += len) {
			for (int k = 0; k < len / 2; k++) {
				double wr = c[k * step], wi = -s[k * step];
				int a = i + k, b = i + k + len / 2;
				double tr = re[b] * wr - im[b] * wi;
				double ti = re[b] * wi + im[b] * wr;
				re[b] = re[a] - tr;
				im[b] = im[a] - ti;
				re[a] += tr;
				im[a] += ti;
			}
		}
	}
}

// Score S of Eq. 5-6.
static double spectral_peak(Tracking *trk)
{
	int n = trk->n_bins;
	double *re = trk->re, *im = trk->im, *counts = trk->counts;
	double sum = 0, noise = 0;
	for (int k = 0; k < n; k++)
		sum += counts[k];
	double mean = sum / n;
	for (int k = 0; k < n; k++) {
		re[k] = (counts[k] - mean) * trk->window[k];
		im[k] = 0;
		noise += counts[k] * (trk->window[k] * trk->window[k]);
	}
	noise = noise > 1.0 ? noise : 1.0;

	double peak = 0;
	if ((n & (n - 1)) == 0) {
		fft(re, im, n, trk->cos_t, trk->sin_t);
		for (int k = trk->k0; k <= trk->k1; k++) {
			double pw = re[k] * re[k] + im[k] * im[k];
			peak = pw > peak ? pw : peak;
		}
	} else {
		// Plain DFT of the band for other sizes.
		for (int k = trk->k0; k <= trk->k1; k++) {
			double xr = 0, xi = 0;
			for (int j = 0; j < n; j++) {
				int a = (int)(((int64_t)j * k) % n);
				xr += re[j] * trk->cos_t[a];
				xi -= re[j] * trk->sin_t[a];
			}
			double pw = xr * xr + xi * xi;
			peak = pw > peak ? pw : peak;
		}
	}
	return peak / noise;
}

// First event of the cell at or after t, searching from lo.
static size_t cell_bound(const Cell *c, size_t lo, int64_t t)
{
	size_t hi = c->len;
	while (lo < hi) {
		size_t mid = lo + (hi - lo) / 2;
		if (c->ev[mid].t < t)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

// True to hide the box: many events but no wingbeat peak.
static bool is_background(Tracking *trk, Tracker *tr, int64_t ts, const int box[4])
{
	const TrackingParams *P = &trk->p;
	int64_t t0 = ts - P->bg_window_us;
	int pad = (int)P->bg_pad;
	int xlo = box[0] - pad > 0 ? box[0] - pad : 0, xhi = box[2] + pad;
	int ylo = box[1] - pad > 0 ? box[1] - pad : 0, yhi = box[3] + pad;
	int cx0 = xlo >> CELL_SHIFT, cx1 = (xhi < P->width ? xhi : P->width - 1) >> CELL_SHIFT;
	int cy0 = ylo >> CELL_SHIFT, cy1 = (yhi < P->height ? yhi : P->height - 1) >> CELL_SHIFT;
	// Count first; cells wholly inside need no check per event.
	int64_t n = 0;
	size_t n_spans = 0;
	for (int cy = cy0; cy <= cy1; cy++) {
		bool rows_in = cy << CELL_SHIFT >= ylo && ((cy + 1) << CELL_SHIFT) - 1 <= yhi;
		for (int cx = cx0; cx <= cx1; cx++) {
			const Cell *c = &trk->cells[cy * trk->n_cx + cx];
			size_t i0 = cell_bound(c, 0, t0), i1 = cell_bound(c, i0, ts);
			trk->spans[n_spans++] = (Span){c, i0, i1};
			if (rows_in && cx << CELL_SHIFT >= xlo && ((cx + 1) << CELL_SHIFT) - 1 <= xhi) {
				n += (int64_t)(i1 - i0);
				continue;
			}
			for (size_t i = i0; i < i1; i++) {
				const Event *e = &c->ev[i];
				n += e->x >= xlo && e->x <= xhi && e->y >= ylo && e->y <= yhi;
			}
		}
	}
	// Bin the event rate only when the spectrum is needed.
	if (n >= P->bg_min_events) {
		int32_t *bins = trk->bins;
		memset(bins, 0, trk->n_bins * sizeof(int32_t));
		for (size_t s = 0; s < n_spans; s++) {
			const Span *sp = &trk->spans[s];
			int64_t bin = 0, bin_end = t0;
			int32_t run = 0;
			for (size_t i = sp->i0; i < sp->i1; i++) {
				const Event *e = &sp->c->ev[i];
				if (e->t >= bin_end) {
					bins[bin] += run;
					run = 0;
					bin = (e->t - t0) / P->bg_bin_us;
					bin_end = t0 + (bin + 1) * P->bg_bin_us;
				}
				run += e->x >= xlo && e->x <= xhi && e->y >= ylo && e->y <= yhi;
			}
			bins[bin] += run;
		}
		for (int k = 0; k < trk->n_bins; k++)
			trk->counts[k] = bins[k];
	}

	double cx = 0.5 * (box[0] + box[2]), cy = 0.5 * (box[1] + box[3]);
	if (tr->background < 0) {
		// A new tracker on an object that was just hidden starts hidden too.
		while (trk->hid_start < trk->hid_end
		       && ts - trk->hidden[trk->hid_start].ts > P->bg_inherit_us)
			trk->hid_start++;
		tr->background = 0;
		for (size_t i = trk->hid_start; i < trk->hid_end; i++) {
			double dx = cx - trk->hidden[i].cx, dy = cy - trk->hidden[i].cy;
			if (dx * dx + dy * dy <= trk->inherit_r2) {
				tr->background = 1;
				break;
			}
		}
	}
	if (n >= P->bg_min_events) {
		double peak = spectral_peak(trk);
		if (peak < P->bg_max_peak)
			tr->background = 1;
		else if (peak >= P->bg_clear_peak)
			tr->background = 0;
	} else if (n < P->bg_small_frac * P->bg_min_events) {
		tr->background = 0;
	}
	if (tr->background) {
		if (trk->hid_end == trk->hid_cap) {
			// Drop the popped part before growing.
			memmove(trk->hidden, trk->hidden + trk->hid_start,
			        (trk->hid_end - trk->hid_start) * sizeof(HiddenBox));
			trk->hid_end -= trk->hid_start;
			trk->hid_start = 0;
			if (trk->hid_end == trk->hid_cap) {
				trk->hid_cap = grown(trk->hid_cap, trk->hid_end + 1);
				trk->hidden = xrealloc(trk->hidden, trk->hid_cap * sizeof(HiddenBox));
			}
		}
		trk->hidden[trk->hid_end++] = (HiddenBox){ts, cx, cy};
	}
	return tr->background;
}

// Keep a non-empty chunk for the background test.
static void keep_events(Tracking *trk, const int32_t *t, const uint16_t *x,
                        const uint16_t *y, size_t n)
{
	trk->keep_from = trk->next_bbox_ts - trk->p.bg_window_us;
	for (size_t i = 0; i < n; i++) {
		Cell *c = &trk->cells[(y[i] >> CELL_SHIFT) * trk->n_cx + (x[i] >> CELL_SHIFT)];
		if (c->len == c->cap) {
			// Drop the events no box can need, then grow if still full.
			size_t old = cell_bound(c, 0, trk->keep_from);
			memmove(c->ev, c->ev + old, (c->len - old) * sizeof(Event));
			c->len -= old;
			if (c->len == c->cap) {
				c->cap = grown(c->cap, c->len + 1);
				c->ev = xrealloc(c->ev, c->cap * sizeof(Event));
			}
		}
		c->ev[c->len++] = (Event){t[i], x[i], y[i]};
	}
	trk->ev_alive = t[n - 1] >= trk->keep_from;
}

/*************************************************************
 * Tracking
 */

static void push_row(BoxList *out, int64_t ts, int id, const int box[4])
{
	if (out->len == out->cap) {
		out->cap = grown(out->cap, out->len + 1);
		out->rows = xrealloc(out->rows, out->cap * sizeof(BoxRow));
	}
	out->rows[out->len++] = (BoxRow){ts, id, box[0], box[1], box[2], box[3]};
}

// Advance the trackers on the readings m_order[0..nm) and emit due rows.
static void step(Tracking *trk, int64_t now, size_t nm, BoxList *out)
{
	const TrackingParams *P = &trk->p;
	int width = P->width;

	// Section III-D, oldest reading first.
	for (size_t j = 0; j < nm; j++) {
		uint32_t pix = trk->m_pix[trk->m_order[j]];
		int32_t ts = trk->px[pix].ts;
		int x = (int)(pix % width), y = (int)(pix / width);
		// Eq. 2.
		double ps = trk->px[pix].period;
		double dp = fabs(ps - trk->p_ideal);
		for (int64_t k = 2; k <= P->max_k; k++)
			dp = fmin(dp, fabs(ps / (double)k - trk->p_ideal));

		// Start.
		if (dp <= P->spawn_max_dp) {
			bool near = false;
			for (size_t i = 0; i < trk->n_tr && !near; i++) {
				double dx = x - trk->tr[i].x_hat, dy = y - trk->tr[i].y_hat;
				near = dx * dx + dy * dy <= trk->spawn_r2;
			}
			if (!near) {
				create_tracker(trk, x, y, ts);
				continue;
			}
		}
		if (trk->n_tr == 0)
			continue;

		// Assignment: Eq. 1-3; the lock bonus only biases the choice.
		size_t best = 0;
		double best_w = 0, best_score = 0;
		for (size_t i = 0; i < trk->n_tr; i++) {
			const Tracker *tr = &trk->tr[i];
			double dx = x - tr->x_hat, dy = y - tr->y_hat;
			double w = P->wmax * exp(-(dx * dx + dy * dy) * trk->inv_sd2
			                         - dp * dp * trk->inv_sp2);
			double score = tr->locked ? w * P->lock_bonus : w;
			if (i == 0 || score > best_score) {
				best = i;
				best_w = w;
				best_score = score;
			}
		}
		if (best_w < P->wmin)
			continue;
		accept(trk, &trk->tr[best], ts, x, y, best_w);
	}

	// One row per locked tracker per step, unless hidden. Before the trim.
	bool gate = P->background_gate && trk->ev_alive;
	while (now >= trk->next_bbox_ts) {
		for (size_t i = 0; i < trk->n_tr; i++) {
			Tracker *tr = &trk->tr[i];
			int box[4];
			if (!get_bbox(trk, tr, trk->next_bbox_ts, box))
				continue;
			if (gate && is_background(trk, tr, trk->next_bbox_ts, box))
				continue;
			push_row(out, trk->next_bbox_ts, tr->id, box);
		}
		trk->next_bbox_ts += trk->interval_us;
	}

	// Drop old matches, then trackers unlocked for T_g (Deletion).
	int64_t cutoff = now - P->bbox_window_us;
	size_t kept = 0;
	for (size_t i = 0; i < trk->n_tr; i++) {
		Tracker *tr = &trk->tr[i];
		while (tr->len && tr->m[tr->head].t < cutoff) {
			tr->head = (tr->head + 1) & (tr->cap - 1);
			tr->len--;
		}
		tr->locked = (int64_t)tr->len >= P->bbox_min_matches;
		if (tr->locked)
			tr->last_locked_ts = now;
		if (now - tr->last_locked_ts < P->unlocked_grace_us)
			trk->tr[kept++] = *tr;
		else
			free(tr->m);
	}
	trk->n_tr = kept;
}

static void reserve_measurements(Tracking *trk, size_t n)
{
	if (n <= trk->m_cap)
		return;
	trk->m_cap = grown(trk->m_cap, n);
	trk->m_pix = xrealloc(trk->m_pix, trk->m_cap * sizeof(uint32_t));
	trk->m_ts = xrealloc(trk->m_ts, trk->m_cap * sizeof(int32_t));
	trk->m_order = xrealloc(trk->m_order, trk->m_cap * sizeof(intptr_t));
}

void tracking_update(Tracking *trk, const int32_t *t, const uint16_t *x,
                     const uint16_t *y, const uint8_t *p, size_t n, BoxList *out)
{
	int width = trk->p.width;
	int64_t last = trk->last_seen, now = last;
	size_t nm = 0;
	reserve_measurements(trk, n + 1);

	// Section III-A: period map. A pixel is listed at its first new period in
	// the chunk; later ones overwrite it.
	for (size_t i = 0; i < n; i++) {
		uint32_t k = (uint32_t)y[i] * width + x[i];
		Pixel *px = &trk->px[k];
		int32_t *prev;
		if (p[i] && px->on2off <= px->off2on)
			prev = &px->on2off;
		else if (!p[i] && px->on2off > px->off2on)
			prev = &px->off2on;
		else
			continue;
		// The first change of each kind gives a negative period.
		px->period = (int32_t)((uint32_t)t[i] - (uint32_t)*prev);
		*prev = t[i];
		if (px->ts <= last)
			trk->m_pix[nm++] = k;
		px->ts = t[i];
		now = t[i] > now ? t[i] : now;
	}
	trk->last_seen = now;

	if (n > 0 && trk->p.background_gate)
		keep_events(trk, t, x, y, n);
	// The first chunk only fills the period map.
	if (!trk->started) {
		trk->started = true;
		return;
	}

	// Drop first periods and periods much longer than a wingbeat.
	size_t kept = 0;
	for (size_t j = 0; j < nm; j++) {
		int32_t period = trk->px[trk->m_pix[j]].period;
		if (period > 0 && period <= trk->p.max_period_us)
			trk->m_pix[kept++] = trk->m_pix[j];
	}
	nm = kept;

	// Oldest first, ties in row-major order.
	sort_pixels(trk->m_pix, (intptr_t)nm);
	for (size_t j = 0; j < nm; j++) {
		trk->m_ts[j] = trk->px[trk->m_pix[j]].ts;
		trk->m_order[j] = (intptr_t)j;
	}
	argsort(trk->m_ts, trk->m_order, (intptr_t)nm);
	step(trk, now, nm, out);
}

void tracking_flush(Tracking *trk, int64_t end_ts, BoxList *out)
{
	step(trk, end_ts, 0, out);
}

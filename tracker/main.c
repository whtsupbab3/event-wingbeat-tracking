// Runs the tracker over an event file and writes the boxes as CSV.
//
//     ./track --config ../configs/default.conf --bbox_csv boxes.csv events.bin
//
// Flags override the config file. Event file (from evaluation/events.py):
// "IWEV", uint32 width, height, 0, uint64 n, then int32 t[n] (us),
// uint16 x[n], uint16 y[n], uint8 p[n], little-endian.
#include "tracking.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
	const char *name;
	char type; // d = double, i = integer, b = boolean
	size_t offset;
} Option;

#define OPT(type, field) {#field, type, offsetof(TrackingParams, field)}
static const Option options[] = {
	OPT('d', target_freq),
	OPT('d', sigma_d),
	OPT('d', sigma_p),
	OPT('d', wmax),
	OPT('d', wmin),
	OPT('i', max_k),
	OPT('i', max_period_us),
	OPT('d', lock_bonus),
	OPT('d', spawn_dist),
	OPT('d', spawn_max_dp),
	OPT('i', unlocked_grace_us),
	OPT('d', bbox_interval_ms),
	OPT('i', bbox_window_us),
	OPT('d', bbox_radius),
	OPT('i', bbox_min_matches),
	OPT('i', bbox_pad),
	OPT('b', background_gate),
	OPT('i', bg_window_us),
	OPT('i', bg_bin_us),
	OPT('i', bg_pad),
	OPT('i', bg_min_events),
	OPT('d', bg_fmin),
	OPT('d', bg_fmax),
	OPT('d', bg_max_peak),
	OPT('d', bg_clear_peak),
	OPT('d', bg_small_frac),
	OPT('d', bg_inherit_px),
	OPT('i', bg_inherit_us),
};
#define N_OPTIONS (sizeof options / sizeof options[0])

static void usage(const char *prog)
{
	fprintf(stderr, "usage: %s [--config FILE] [--dt SECONDS] [--bbox_csv FILE] "
	                "[--<param> VALUE ...] EVENTS.bin\n"
	                "params (booleans also as --<param> / --no-<param>):", prog);
	for (size_t i = 0; i < N_OPTIONS; i++)
		fprintf(stderr, " %s", options[i].name);
	fprintf(stderr, "\n");
	exit(2);
}

static bool parse_bool(const char *value, bool *out)
{
	if (strcmp(value, "true") == 0 || strcmp(value, "1") == 0) {
		*out = true;
		return true;
	}
	if (strcmp(value, "false") == 0 || strcmp(value, "0") == 0) {
		*out = false;
		return true;
	}
	return false;
}

// Sets a parameter, or dt (chunk size in seconds); false if invalid.
static bool set_param(TrackingParams *params, double *dt, const char *name, const char *value)
{
	char *end;
	if (strcmp(name, "dt") == 0) {
		*dt = strtod(value, &end);
		return end != value && *end == '\0' && *dt > 0;
	}
	for (size_t k = 0; k < N_OPTIONS; k++) {
		const Option *o = &options[k];
		if (strcmp(name, o->name) != 0)
			continue;
		char *field = (char *)params + o->offset;
		if (o->type == 'b')
			return parse_bool(value, (bool *)field);
		if (o->type == 'd')
			*(double *)field = strtod(value, &end);
		else
			*(int64_t *)field = strtoll(value, &end, 10);
		return end != value && *end == '\0';
	}
	return false;
}

static char *trim(char *s)
{
	while (isspace((unsigned char)*s))
		s++;
	char *e = s + strlen(s);
	while (e > s && isspace((unsigned char)e[-1]))
		*--e = '\0';
	return s;
}

// Reads "name = value" lines; "#" starts a comment.
static void read_config(const char *path, TrackingParams *params, double *dt)
{
	FILE *f = fopen(path, "r");
	if (f == NULL) {
		fprintf(stderr, "could not read %s\n", path);
		exit(1);
	}
	char line[512];
	for (int n = 1; fgets(line, sizeof line, f) != NULL; n++) {
		char *hash = strchr(line, '#');
		if (hash != NULL)
			*hash = '\0';
		char *s = trim(line);
		if (*s == '\0')
			continue;
		char *eq = strchr(s, '=');
		if (eq == NULL) {
			fprintf(stderr, "%s:%d: expected name = value\n", path, n);
			exit(1);
		}
		*eq = '\0';
		char *name = trim(s), *value = trim(eq + 1);
		if (!set_param(params, dt, name, value)) {
			fprintf(stderr, "%s:%d: bad parameter %s = %s\n", path, n, name, value);
			exit(1);
		}
	}
	fclose(f);
}

static void *read_array(FILE *f, size_t n, size_t size)
{
	void *a = malloc(n * size + 1);
	if (a == NULL || fread(a, size, n, f) != n) {
		fprintf(stderr, "could not read the event file\n");
		exit(1);
	}
	return a;
}

int main(int argc, char **argv)
{
	double dt = 0.01;
	const char *bbox_csv = NULL, *source = NULL;
	TrackingParams params;
	tracking_default_params(&params, 0, 0);

	for (int i = 1; i + 1 < argc; i++)
		if (strcmp(argv[i], "--config") == 0)
			read_config(argv[i + 1], &params, &dt);
	for (int i = 1; i < argc;) {
		const char *arg = argv[i];
		if (strcmp(arg, "--config") == 0 && i + 1 < argc) {
			i += 2;
		} else if (strcmp(arg, "--bbox_csv") == 0 && i + 1 < argc) {
			bbox_csv = argv[i + 1];
			i += 2;
		} else if (strncmp(arg, "--no-", 5) == 0) {
			if (!set_param(&params, &dt, arg + 5, "false"))
				usage(argv[0]);
			i += 1;
		} else if (strncmp(arg, "--", 2) == 0) {
			if (i + 1 < argc && set_param(&params, &dt, arg + 2, argv[i + 1])) {
				i += 2;
			} else if (set_param(&params, &dt, arg + 2, "true")) {
				i += 1;
			} else {
				fprintf(stderr, "bad option %s\n", arg);
				usage(argv[0]);
			}
		} else if (source == NULL) {
			source = argv[i++];
		} else {
			usage(argv[0]);
		}
	}
	if (source == NULL)
		usage(argv[0]);

	FILE *f = fopen(source, "rb");
	char magic[4];
	uint32_t header[3];
	uint64_t n;
	if (f == NULL || fread(magic, 1, 4, f) != 4 || memcmp(magic, "IWEV", 4) != 0
	    || fread(header, 4, 3, f) != 3 || fread(&n, 8, 1, f) != 1 || n == 0) {
		fprintf(stderr, "%s is not an event file from evaluation/events.py\n", source);
		return 1;
	}
	params.width = (int)header[0];
	params.height = (int)header[1];
	int32_t *t = read_array(f, n, sizeof(int32_t));
	uint16_t *x = read_array(f, n, sizeof(uint16_t));
	uint16_t *y = read_array(f, n, sizeof(uint16_t));
	uint8_t *p = read_array(f, n, sizeof(uint8_t));
	fclose(f);
	for (uint64_t i = 0; i < n; i++) {
		if (x[i] >= params.width || y[i] >= params.height) {
			fprintf(stderr, "event %llu at (%d, %d) is outside the %dx%d sensor\n",
			        (unsigned long long)i, x[i], y[i], params.width, params.height);
			return 1;
		}
		if (i > 0 && t[i] < t[i - 1]) {
			fprintf(stderr, "event %llu is out of time order\n", (unsigned long long)i);
			return 1;
		}
	}

	// Timing covers processing only.
	Tracking *trk = tracking_new(&params);
	BoxList rows = {0};
	int64_t step_us = (int64_t)nearbyint(params.bbox_interval_ms * 1000);
	int32_t chunk = (int32_t)nearbyint(dt * 1e6);

	struct timespec tic, toc;
	clock_gettime(CLOCK_MONOTONIC, &tic);
	int32_t now = t[0];
	size_t start = 0;
	while (start < n) {
		now += chunk;
		// First event at or after now, as np.searchsorted.
		size_t lo = start, hi = n;
		while (lo < hi) {
			size_t mid = lo + (hi - lo) / 2;
			if (t[mid] < now)
				lo = mid + 1;
			else
				hi = mid;
		}
		tracking_update(trk, t + start, x + start, y + start, p + start, lo - start, &rows);
		start = lo;
	}
	tracking_flush(trk, (t[n - 1] + step_us - 1) / step_us * step_us, &rows);
	clock_gettime(CLOCK_MONOTONIC, &toc);
	double busy = (double)(toc.tv_sec - tic.tv_sec) + 1e-9 * (double)(toc.tv_nsec - tic.tv_nsec);

	if (bbox_csv != NULL) {
		FILE *out = fopen(bbox_csv, "w");
		if (out == NULL) {
			fprintf(stderr, "could not write %s\n", bbox_csv);
			return 1;
		}
		fprintf(out, "timestamp,track_id,x1,y1,x2,y2\n");
		for (size_t i = 0; i < rows.len; i++) {
			const BoxRow *r = &rows.rows[i];
			fprintf(out, "%lld,%d,%d,%d,%d,%d\n", (long long)r->ts, r->id,
			        r->x1, r->y1, r->x2, r->y2);
		}
		fclose(out);
		printf("wrote %zu boxes to %s\n", rows.len, bbox_csv);
	} else {
		printf("%zu boxes, %d trackers alive at the end\n", rows.len, tracking_num_trackers(trk));
	}
	printf("proc speed=%.0f%% of real time\n", (t[n - 1] - t[0]) * 1e-6 / busy * 100);
	printf("busy_s=%.6f\n", busy);

	tracking_free(trk);
	free(rows.rows);
	free(t);
	free(x);
	free(y);
	free(p);
	return 0;
}

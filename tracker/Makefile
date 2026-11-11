.PHONY: all clean
CFLAGS = -std=c11 -O3 -Wall -Wextra -pedantic -D_DEFAULT_SOURCE=1
CFLAGS += -ffp-contract=off

all: track
track: main.c tracking.c tracking.h
	$(CC) $(CFLAGS) main.c tracking.c -o $@ -lm
clean:
	rm -f track

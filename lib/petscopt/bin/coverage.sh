#!/bin/sh
set -eu

export PETSCOPT_ARCH=arch-coverage
export CFLAGS=--coverage
export FFLAGS=--coverage

make distclean
make config
make
make test

SRCDIR=src
OBJDIR=$PETSCOPT_ARCH/obj
OUTDIR=$PETSCOPT_ARCH

gcovr \
    -j 4 \
    --exclude-unreachable-branches \
    --print-summary \
    --html-details \
    --html-title "PETSCOPT Coverage" \
    --sort-percentage \
    --root $SRCDIR \
    --output $OUTDIR/index.html \
    $OBJDIR

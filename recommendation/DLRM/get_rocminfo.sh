#!/bin/bash

sudo /opt/rocm/bin/rocminfo | \
  grep gfx90 | sed 's/.*\(gfx90.\).*/\1/g' | sort | uniq

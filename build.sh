#!/bin/sh

mkdir build || true
cd build && cmake .. -DSD_METAL=ON && cmake --build . --config Release

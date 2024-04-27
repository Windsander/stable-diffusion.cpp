#!/bin/sh

mkdir build || true
cd build && cmake .. && cmake --build . --config Release

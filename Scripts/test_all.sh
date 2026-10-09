#!/bin/sh
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
# Debug suite (fast; real-model tests skip) then optimised suite (runs real-model validation if ./Models is populated).
set -eu
cd "$(dirname "$0")/.."
swift test
swift test -c release -Xswiftc -enable-testing

#!/usr/bin/env bash
# Checks out the Qiskit Aer revision that CMakeLists.txt pins (the qiskit-aer
# FetchContent GIT_TAG), so every build that takes Aer from a directory
# (AER_INCLUDE_DIR) compiles the revision CI tests: the CI and wheel workflows
# and build.sh.
#
#   fetch_pinned_aer.sh --print         print the pinned revision
#   fetch_pinned_aer.sh <directory>     make <directory> a checkout of it
#
# An existing checkout is moved to the pinned revision unless it is there
# already; a directory that is not a checkout is turned into one.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repository="https://github.com/QoroQuantum/qiskit-aer.git"
revision="$(awk '/GIT_REPOSITORY https:\/\/github.com\/QoroQuantum\/qiskit-aer.git/ {getline; print $2; exit}' "$root/CMakeLists.txt" | tr -d '\r')"
if ! [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "No qiskit-aer GIT_TAG in $root/CMakeLists.txt" >&2
    exit 1
fi

if [ "${1:-}" = "--print" ]; then
    echo "$revision"
    exit 0
fi

destination="${1:?usage: fetch_pinned_aer.sh --print | <directory>}"
if [ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" = "$revision" ]; then
    exit 0
fi
mkdir -p "$destination"
[ -d "$destination/.git" ] || git init -q "$destination"
git -C "$destination" fetch -q --depth 1 "$repository" "$revision"
git -C "$destination" checkout -q FETCH_HEAD

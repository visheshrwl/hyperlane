#!/usr/bin/env bash
# Packages a built shardline-worker as a release tarball: the binary and the licenses
# of what is linked into it. On macOS it also bundles the Homebrew libraries the binary
# loads (gRPC, protobuf, abseil, ...) and rewrites them to load from the tarball, so
# the worker runs without Homebrew and survives Homebrew upgrades.
#
#   scripts/package_worker.sh BUILD_DIR VERSION PLATFORM OUT_DIR
#
# Written for macOS's bash 3.2.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BUILD_DIR VERSION PLATFORM OUT_DIR" >&2
  exit 2
fi
build=$1 version=$2 platform=$3 out=$4
root="$(cd "$(dirname "$0")/.." && pwd)"
name="shardline-worker-$version-$platform"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
stage="$work/$name"

mkdir -p "$stage/bin" "$stage/licenses"
cp "$build/shardline-worker" "$stage/bin/"
cp "$root/LICENSE" "$stage/LICENSE"
cp "$build/_deps/llama_cpp-src/LICENSE" "$stage/licenses/ggml.txt"

# The libraries a Mach-O file links that live outside the system, one name per line,
# as the file refers to them.
external_refs() {
  otool -L "$1" | tail -n +2 | awk '{print $1}' |
    grep -E '^(/opt/homebrew/|/usr/local/|@rpath/|@loader_path/)' |
    grep -vxF "$(otool -D "$1" | tail -n +2)" || true
}

# Where a reference from SOURCE (the original, unbundled file) resolves on this machine.
resolve() {
  case $2 in
    @rpath/* | @loader_path/*) echo "$(dirname "$(realpath "$1")")/${2#*/}" ;;
    *) echo "$2" ;;
  esac
}

if [[ $(uname) == Darwin ]]; then
  mkdir -p "$stage/lib"
  # Pairs of "bundled file<TAB>file it was copied from", processed breadth first.
  printf '%s\t%s\n' "$stage/bin/shardline-worker" "$build/shardline-worker" >"$work/queue"
  done_count=0
  while [[ $done_count -lt $(wc -l <"$work/queue") ]]; do
    done_count=$((done_count + 1))
    line=$(sed -n "${done_count}p" "$work/queue")
    file=${line%%$'\t'*} source=${line#*$'\t'}
    for ref in $(external_refs "$file"); do
      base=$(basename "$ref")
      if [[ ! -e $stage/lib/$base ]]; then
        lib=$(realpath "$(resolve "$source" "$ref")")
        cp "$lib" "$stage/lib/$base"
        chmod u+w "$stage/lib/$base"
        install_name_tool -id "@rpath/$base" "$stage/lib/$base" 2>/dev/null
        printf '%s\t%s\n' "$stage/lib/$base" "$lib" >>"$work/queue"
        # Ship the license of the Homebrew formula the library came from.
        formula=$(echo "$lib" | sed -nE 's|.*/Cellar/([^/]+)/.*|\1|p')
        if [[ -z $formula ]]; then
          echo "$lib: not from a Homebrew formula; cannot find its license" >&2
          exit 1
        fi
        if [[ ! -d $stage/licenses/$formula ]]; then
          mkdir "$stage/licenses/$formula"
          prefix=${lib%%/Cellar/*}/opt/$formula
          find "$prefix/" -maxdepth 1 \( -iname 'licen[cs]e*' -o -iname 'copying*' \) \
            -exec cp {} "$stage/licenses/$formula/" \;
          if [[ -z $(ls -A "$stage/licenses/$formula") ]]; then
            echo "$prefix: no license file" >&2
            exit 1
          fi
        fi
      fi
      install_name_tool -change "$ref" "@rpath/$base" "$file" 2>/dev/null
    done
    # Drop search paths into Homebrew, so an installed Homebrew cannot shadow a bundled library.
    for rpath in $(otool -l "$file" | awk '/cmd LC_RPATH/{getline; getline; print $2}'); do
      install_name_tool -delete_rpath "$rpath" "$file" 2>/dev/null
    done
  done
  install_name_tool -add_rpath @executable_path/../lib "$stage/bin/shardline-worker" 2>/dev/null
  for f in "$stage/lib/"*; do install_name_tool -add_rpath @loader_path "$f" 2>/dev/null; done
  # Rewriting load commands invalidates the signatures; arm64 refuses unsigned code.
  for f in "$stage/lib/"* "$stage/bin/shardline-worker"; do codesign --force --sign - "$f" 2>/dev/null; done

  leftover=$(for f in "$stage/lib/"* "$stage/bin/shardline-worker"; do
    otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -E '^(/opt/homebrew/|/usr/local/)' || true
  done)
  if [[ -n $leftover ]]; then
    echo "still linked outside the bundle:" >&2
    echo "$leftover" | sort -u >&2
    exit 1
  fi
  echo "bundled $(find "$stage/lib" -type f | wc -l | tr -d ' ') libraries"
else
  echo "runtime libraries:"
  ldd "$stage/bin/shardline-worker"
fi

mkdir -p "$out"
tar -C "$work" -czf "$out/$name.tar.gz" "$name"
echo "$out/$name.tar.gz"

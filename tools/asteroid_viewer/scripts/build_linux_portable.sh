#!/bin/sh
# Builds bin/asteroid_viewer for Linux x86-64 that runs on any distribution with glibc 2.27 or newer
# (Ubuntu 18.04+, Debian 10+, RHEL/Rocky 8+, Fedora, ...). The C++ runtime is linked in; at run time it needs
# only libc and the system's X11 and OpenGL libraries (opened by GLFW).
#
# Needs: cmake, git, zig (e.g. `pip install ziglang`), and the X11 headers
# (Debian/Ubuntu: libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev).
#
#   scripts/build_linux_portable.sh [extra cmake options, e.g. -DPERIOD_SEARCH_DIR=...]
set -e
cd "$(dirname "$0")/.."
SRC=$(pwd)
OUT=$SRC/build-linux
mkdir -p "$OUT/zig"

if command -v zig >/dev/null 2>&1; then ZIG="zig"; else ZIG="python3 -m ziglang"; fi
GLIBC=2.27

# zig's own glibc headers must not be mixed with the host's /usr/include: copy the X11 headers aside
mkdir -p "$OUT/zig/inc"
cp -r /usr/include/X11 "$OUT/zig/inc/"
for t in cc c++ ar ranlib; do
  case $t in cc|c++) extra="-target x86_64-linux-gnu.$GLIBC -isystem $OUT/zig/inc" ;; *) extra="" ;; esac
  printf '#!/bin/sh\nexec %s %s %s "$@"\n' "$ZIG" "$t" "$extra" > "$OUT/zig/z$t"
  chmod +x "$OUT/zig/z$t"
done
cat > "$OUT/zig/toolchain.cmake" <<EOF
set(CMAKE_C_COMPILER "$OUT/zig/zcc")
set(CMAKE_CXX_COMPILER "$OUT/zig/zc++")
set(CMAKE_AR "$OUT/zig/zar")
set(CMAKE_RANLIB "$OUT/zig/zranlib")
EOF

# GLFW's FindX11: point every header check at the copied headers (the libraries are opened at run time)
X11ARGS=""
for c in X11 Xlib Xutil Xkb Xkblib Xrandr Xinerama Xcursor Xi Xshape Xext Xrender Xfixes; do
  X11ARGS="$X11ARGS -DX11_${c}_INCLUDE_PATH=$OUT/zig/inc"
done
X11LIB=$(ls /usr/lib/*/libX11.so /usr/lib64/libX11.so /usr/lib/libX11.so 2>/dev/null | head -1)

cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$OUT/zig/toolchain.cmake" \
  -DX11_X11_LIB="$X11LIB" $X11ARGS "$@"
cmake --build "$OUT" -j"$(nproc)"
mkdir -p "$SRC/bin"
cp "$OUT/asteroid_viewer" "$SRC/bin/asteroid_viewer"
strip "$SRC/bin/asteroid_viewer" 2>/dev/null || true
echo "built $SRC/bin/asteroid_viewer"

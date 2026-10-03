#!/bin/bash
# Regenerate period_search/kernels.cpp from the 14 live .cl files, matching the
# host concatenation order in Start_OpenCl.cpp (constants.h .. Start.cl) and the
# stock stringify.bat convention (input basename "kernelSource" -> symbol
# ocl_src_kernelSource). After running this you must rebuild the host and delete
# kernels.bin so the edited kernels reach the GPU.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/period_search"
ORDER="constants.h GlobalsCL.h Intrinsics.cl swap.cl blmatrix.cl curv.cl Curv2.cl bright.cl conv.cl mrqcof.cl gauss_errc.cl mrqmin.cl Start.cl"
cat $ORDER > "$ROOT/kernelSource.cl"
cd "$ROOT"
python3 oclProgramFileToString.py kernelSource.cl period_search/kernels.cpp
head -1 period_search/kernels.cpp

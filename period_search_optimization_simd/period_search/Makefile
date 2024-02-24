# This should work on Linux.  Modify as needed for other platforms.

BOINC_DIR = ../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib

CXXFLAGS = -march=x86-64 -O3 -std=c++17 -w -s\
	-DAPP_GRAPHICS \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-L /usr/X11R6/lib \
	-L.

CC=g++
CXX=g++
OPTFLAGS=-march=x86-64 -O3 -std=c++17
CFLAGS=$(OPTFLAGS) -w -s
LDFLAGS=-lm
MY_LIBS=areanorm.o arrayHelpers.o blmatrix.o bright.o bright_avx.o bright_avx512.o bright_fma.o \
    bright_sse2.o bright_sse3.o conv.o conv_avx.o conv_avx512.o conv_fma.o conv_sse2.o \
    conv_sse3.o covsrt.o CpuInfo.o curv.o curv_sse3.o curv_avx_fma.o curv_avx512.o \
    curv_sse2.o dot_product.o ellfit.o EnumHelpers.o gauss_errc.o gauss_errc_avx_fma.o \
    gauss_errc_avx512.o gauss_errc_sse2.o gauss_errc_sse3.o lubksb.o ludcmp.o matrix.o \
    memory.o mrqcof.o mrqcof_avx.o mrqcof_avx512.o mrqcof_fma.o mrqcof_sse2.o \
    mrqcof_sse3.o mrqmin.o phasec.o sphfunc.o systeminfo.o \
    trifac.o VersionInfo.o

# the following should be freeglut; use nm to check
# you may have to change the paths for your system

ifeq ($(wildcard /usr/local/lib/libglut.a),)
	LIBGLUT = /usr/lib/libglut.a
	LIBGLU = /usr/lib/libGLU.a
	LIBJPEG = /usr/lib/libjpeg.a
else
	LIBGLUT = /usr/local/lib/libglut.a
	LIBGLU = /usr/local/lib/libGLU.a
	LIBJPEG = /usr/local/lib/libjpeg.a
endif

PROGS = period_search_BOINC_linux_10215_x64_universal_linux_Release
# uc2_graphics
# make this optional so compile doesn't break on systems without OpenGL

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

clean:
	/bin/rm -f $(PROGS) *.o libstdc++.a

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a

install: period_search_BOINC_linux_10215_x64_universal_linux_Release

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.

#uc2: uc2.o libstdc++.a $(BOINC_API_DIR)/libboinc_api.a #$(BOINC_LIB_DIR)/libboinc.a
#	$(CXX) $(CXXFLAGS) -o uc2 uc2.o libstdc++.a -pthread \
#	$(BOINC_API_DIR)/libboinc_api.a \
#	$(BOINC_LIB_DIR)/libboinc.a

period_search_BOINC_linux_10215_x64_universal_linux_Release: period_search_BOINC.o $(MY_LIBS) libstdc++.a $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a
	$(CXX) $(CXXFLAGS) -o $@ $(MY_LIBS) $< libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a \
	$(BOINC_LIB_DIR)/libboinc.a -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

# This should work on Linux.  Modify as needed for other platforms.

BOINC_DIR = ../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib

CXXFLAGS = -march=corei7-avx -O3 -mavx -mfma -DFMA -std=c++17 -w -s\
	-DAPP_GRAPHICS \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-L /usr/X11R6/lib \
	-L.

CC=g++
CXX=g++
OPTFLAGS=-march=corei7-avx -O3 -mavx -mfma -std=c++17
CFLAGS=$(OPTFLAGS) -w -DFMA -s
LDFLAGS=-lm
MY_LIBS=trifac.o areanorm.o sphfunc.o ellfit.o ludcmp.o lubksb.o mrqmin.o mrqcof_fma.o\
		curv.o blmatrix.o conv_fma.o covsrt.o phasec.o matrix.o bright_fma.o memory.o\
	dot_product.o gauss_errc_fma.o VersionInfo.o CpuInfo.o TargetInstructionSet.o

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

PROGS = period_search_BOINC_linux_10215_x64_fma_linux_Release
# uc2_graphics
# make this optional so compile doesn't break on systems without OpenGL

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

clean:
	/bin/rm -f $(PROGS) *.o libstdc++.a

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a

install: period_search_BOINC_linux_10215_x64_fma_linux_Release

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.

#uc2: uc2.o libstdc++.a $(BOINC_API_DIR)/libboinc_api.a #$(BOINC_LIB_DIR)/libboinc.a
#	$(CXX) $(CXXFLAGS) -o uc2 uc2.o libstdc++.a -pthread \
#	$(BOINC_API_DIR)/libboinc_api.a \
#	$(BOINC_LIB_DIR)/libboinc.a

period_search_BOINC_linux_10215_x64_fma_linux_Release: period_search_BOINC.cpp $(MY_LIBS) libstdc++.a $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a
	$(CXX) $(CXXFLAGS) -o $@ $(MY_LIBS) $< libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a \
	$(BOINC_LIB_DIR)/libboinc.a -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

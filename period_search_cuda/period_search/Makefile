# This should work on Linux.  Modify as needed for other platforms.

BOINC_DIR = ../..
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
CUDA_PATH = /usr/local/cuda-10.2

CXXFLAGS = -g \
    -I$(BOINC_DIR) \
    -I$(BOINC_LIB_DIR) \
    -I$(BOINC_API_DIR) \
    -I$(CUDA_PATH)/include \
    -L /usr/X11R6/lib \
    -L.

# There were two cases from the legacy code - first was using --maxrregcount=80 which is very ineficient on block occupancy.
# After precise wxamination on results from NVIDIA Nsight Performance Analysis both are set to 32 now. This way we achieve 100% block occupancy (most of the time).

# CUDA 5.5: NVCCFLAGS = --ptxas-options=-v --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_20,code=sm_21 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_35,code=compute_35 --maxrregcount=80 -rdc=true
# CUDA 10.2 - CC: compute_30,sm_30;compute_35,sm_35;compute_37,sm_37;compute_50,sm_50;compute_52,sm_52;compute_60,sm_60;compute_61,sm_61;compute_70,sm_70;compute_75,sm_75
NVCCFLAGS =    --ptxas-options=-v --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_37,code=sm_37 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --maxrregcount=32 -rdc=true
NVCCFLAGScv2 = --ptxas-options=-v --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_37,code=sm_37 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --maxrregcount=32 -rdc=true

CC=g++
CXX=g++
OPTFLAGS=-O3
CFLAGS=$(OPTFLAGS) -Wall
LDFLAGS=-lm -lrt -Xlinker -rpath .


MY_LIBS=trifac.o areanorm.o sphfunc.o ellfit.o ludcmp.o lubksb.o mrqmin.o mrqcof.o\
        curv.o blmatrix.o conv.o covsrt.o matrix.o bright.o memory.o\
	dot_product.o gauss_errc.o pscuda.device-link.o start_CUDA.o curve2_CUDA.o Start.o

PROGS = period_search_BOINC_cuda102

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

libcudart.so.10.2:
	ln -s $(CUDA_PATH)/lib64/libcudart.so.10.2 libcudart.so.10.2

clean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcudart.so.10.2

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcudart.so.10.2

install: period_search_BOINC_cuda102

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.

period_search_BOINC_cuda102: period_search_BOINC.o $(MY_LIBS) libstdc++.a $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a libcudart.so.10.2
	$(CXX) $(CXXFLAGS) -o $@ $(MY_LIBS) $< libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a libcudart.so.10.2 \
	$(BOINC_LIB_DIR)/libboinc.a $(LDFLAGS)

curve2_CUDA.o: curve2_CUDA.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGScv2) \
		--compiler-options="$(CXXFLAGS)" \
		--compile curve2_CUDA.cu

start_CUDA.o: start_CUDA.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile start_CUDA.cu

Start.o: Start.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile Start.cu

blmatrix.o: blmatrix.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile blmatrix.cu

bright.o: bright.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile bright.cu

conv.o: conv.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile conv.cu

curv.o: curv.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile curv.cu

gauss_errc.o: gauss_errc.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile gauss_errc.cu

mrqcof.o: mrqcof.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile mrqcof.cu

mrqmin.o: mrqmin.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile mrqmin.cu

matrix.o: matrix.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile matrix.cu

pscuda.device-link.o: start_CUDA.o Start.o blmatrix.o bright.o conv.o curv.o gauss_errc.o mrqcof.o mrqmin.o matrix.o curve2_CUDA.o
	$(CUDA_PATH)/bin/nvcc -dlink -o pscuda.device-link.o --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
    start_CUDA.o Start.o blmatrix.o bright.o conv.o curv.o gauss_errc.o mrqcof.o mrqmin.o matrix.o curve2_CUDA.o

%.o: %.c
	$(CC) $(CFLAGS) -c $<

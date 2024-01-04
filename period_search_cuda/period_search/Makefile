# This should work on Linux.  Modify as needed for other platforms.

# READ ME FIRST:
#
# To keep constinstance of the repository and not be relying to a hardcoded paths please create symbolic links that will be outside the path under
# versioning controll and this way will not brake anyones environment setups.
# From your /home/[YOUR_USERNAME] direcory run: 
#	ln -s /usr/local/cuda-12.0 cuda-12.0
#	ln -s /[your_bpinc_path] boinc_src 
# or change those paths acordingly to your setup.
#

# DO-NOT-CHANGE!
BOINC_DIR = ../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
CUDA_PATH = ../../../cuda-12.0
# END DO-NOT-CHANGE

CXXFLAGS = -g -O3 \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-I$(CUDA_PATH)/include \
	-L /usr/X11R6/lib \
	-L.

# There were two cases from the legacy code - first was using --maxrregcount=80 which is very ineficient on block occupancy.
# After precise examination on results from NVIDIA Nsight Performance Analysis both are set to 32 now. This way we achieve 100% block occupancy (most of the time).
# 28 APR 2020 - Temporarily '--maxrregcount' is set back to '80' for the NVCCFLAGS until further research proves it needs to be set back to '32' 

# CUDA 5.5: NVCCFLAGS = --ptxas-options=-v --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_20,code=sm_21 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_35,code=compute_35 --maxrregcount=80 -rdc=true
# CUDA 10.2 - CC: NVCCFLAGS = compute_30,sm_30;compute_35,sm_35;compute_37,sm_37;compute_50,sm_50;compute_52,sm_52;compute_60,sm_60;compute_61,sm_61;compute_70,sm_70;compute_75,sm_75
# CUDA 11.1 - CC: NVCCFLAGS = compute_52,sm_52;compute_60,sm_60;compute_61,sm_61;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86
# CUDA 11.8 - CC: NVCCFLAGS = --ptxas-options=-v --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --maxrregcount=80 -rdc=tru
# CUDA 12.0 - CC:
NVCCFLAGS = --forward-unknown-opts -mcmodel=medium --split-compile-extended 25 -O3 -lineinfo --use_fast_math --extra-device-vectorization --ptxas-options="-v -O3 --register-usage-level 0 " -t0 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_89,code=sm_89 --maxrregcount=32 -rdc=true

CC=g++
CXX=g++
OPTFLAGS=-O3
CFLAGS=$(OPTFLAGS) -Wall -mcmodel=medium
LDFLAGS=-lm -lrt -lstdc++ -lcudart -lcuda -Xlinker -rpath -mcmodel=medium .


MY_LIBS=trifac.o areanorm.o sphfunc.o ellfit.o ludcmp.o lubksb.o \
		covsrt.o memory.o\
		dot_product.o pscuda.device-link.o start_CUDA.o Start.o\
		ComputeCapability.o

PROGS = period_search_BOINC_cuda12000

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

libcudart.so:
	ln -s $(CUDA_PATH)/lib64/libcudart.so libcudart.so

libcuda.so:
	ln -s $(CUDA_PATH)/lib64/stubs/libcuda.so libcuda.so

libnvidia-ml.so:
	ln -s $(CUDA_PATH)/lib64/stubs/libnvidia-ml.so libnvidia-ml.so

clean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcuda.so libcudart.so libnvidia-ml.so

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcuda.so libcudart.so libnvidia-ml.so

install: period_search_BOINC_cuda12000

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.
period_search_BOINC_cuda12000: period_search_BOINC.o $(MY_LIBS) $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a libstdc++.a libcudart.so libcuda.so libnvidia-ml.so
	$(CXX) $(CXXFLAGS) -o $@ $(MY_LIBS) $< libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a \
	$(BOINC_LIB_DIR)/libboinc.a $(LDFLAGS) \
	libcudart.so \
	libcuda.so \
	libnvidia-ml.so

ComputeCapability.o: ComputeCapability.cu
	$(CUDA_PATH)/bin/nvcc --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
		--compile ComputeCapability.cu

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

pscuda.device-link.o: start_CUDA.o Start.o ComputeCapability.o
	$(CUDA_PATH)/bin/nvcc -dlink -o pscuda.device-link.o --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
	start_CUDA.o Start.o ComputeCapability.o

%.o: %.c
	$(CC) $(CFLAGS) -c $<

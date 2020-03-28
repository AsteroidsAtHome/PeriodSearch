# This should work on Linux.  Modify as needed for other platforms.

BOINC_DIR = ../..
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
CUDA_PATH = /usr/local/cuda-5.5

CXXFLAGS = -g \
    -I$(BOINC_DIR) \
    -I$(BOINC_LIB_DIR) \
    -I$(BOINC_API_DIR) \
    -I$(CUDA_PATH)/include \
    -L /usr/X11R6/lib \
    -L.

NVCCFLAGS = --ptxas-options=-v --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_20,code=sm_21 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_35,code=compute_35 --maxrregcount=80 -rdc=true
NVCCFLAGScv2 = --ptxas-options=-v --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_20,code=sm_21 --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_35,code=compute_35 --maxrregcount=32 -rdc=true

CC=g++
CXX=g++
OPTFLAGS=-O3
CFLAGS=$(OPTFLAGS) -Wall 
LDFLAGS=-lm -lrt -Xlinker -rpath .


MY_LIBS=trifac.o areanorm.o sphfunc.o ellfit.o ludcmp.o lubksb.o mrqmin.o mrqcof.o\
        curv.o blmatrix.o conv.o covsrt.o matrix.o bright.o memory.o\
	dot_product.o gauss_errc.o pscuda.device-link.o start_CUDA.o curve2_CUDA.o

PROGS = period_search_BOINC_cuda 

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

libcudart.so.5.5:  
	ln -s $(CUDA_PATH)/lib64/libcudart.so.5.5 libcudart.so.5.5

clean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcudart.so.5.5

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a libcudart.so.5.5

install: period_search_BOINC_cuda

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.

period_search_BOINC_cuda: period_search_BOINC.o $(MY_LIBS) libstdc++.a $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a libcudart.so.5.5
	$(CXX) $(CXXFLAGS) -o $@ $(MY_LIBS) $< libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a libcudart.so.5.5 \
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

pscuda.device-link.o: start_CUDA.o blmatrix.o bright.o conv.o curv.o gauss_errc.o mrqcof.o mrqmin.o matrix.o curve2_CUDA.o
	$(CUDA_PATH)/bin/nvcc -dlink -o pscuda.device-link.o --compiler-bindir=/usr/bin/gcc $(OPTIMIZATION) \
		$(NVCCFLAGS) \
		--compiler-options="$(CXXFLAGS)" \
    start_CUDA.o blmatrix.o bright.o conv.o curv.o gauss_errc.o mrqcof.o mrqmin.o matrix.o curve2_CUDA.o
    
%.o: %.c
	$(CC) $(CFLAGS) -c $<

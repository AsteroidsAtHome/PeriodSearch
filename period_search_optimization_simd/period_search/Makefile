# This should work on Linux.  Modify as needed for other platforms.

# Define directories
BUILD_DIR = build
BOINC_DIR = ../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
PS_COMMON_DIR = ../../period_search_common
PS_COMMON_LIB_DIR = $(PS_COMMON_DIR)/build

# Keep it commented out for the gitactions, but it's handy for local builds
#MAKEFLAGS += -j8

# Define compiler and flags
CXX = g++
CXXFLAGS = -march=x86-64 -O3 -std=c++2a -w -s \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-I$(PS_COMMON_DIR) \
	-L$(PS_COMMON_LIB_DIR) \
	-L/usr/X11R6/lib \
	-L.

# Define the target executable
TARGET = $(BUILD_DIR)/period_search_BOINC_linux_10222_x64_universal_Release

# Define the source files
SRC = areanorm.cpp blmatrix.cpp bright.cpp bright_avx.cpp bright_avx512.cpp bright_fma.cpp \
	bright_sse2.cpp bright_sse3.cpp conv.cpp conv_avx.cpp conv_avx512.cpp conv_fma.cpp conv_sse2.cpp \
	conv_sse3.cpp covsrt.cpp CpuInfo.cpp curv.cpp curv_sse3.cpp curv_avx.cpp curv_fma.cpp curv_avx512.cpp \
	curv_sse2.cpp ellfit.cpp EnumHelpers.cpp gauss_errc.cpp gauss_errc_avx.cpp gauss_errc_fma.cpp \
	gauss_errc_avx512.cpp gauss_errc_sse2.cpp gauss_errc_sse3.cpp lubksb.cpp ludcmp.cpp matrix.cpp \
	mrqcof.cpp mrqcof_avx.cpp mrqcof_avx512.cpp mrqcof_fma.cpp mrqcof_sse2.cpp \
	mrqcof_sse3.cpp mrqmin.cpp phasec.cpp sphfunc.cpp systeminfo.cpp \
	trifac.cpp VersionInfo.cpp benchmark.cpp period_search_BOINC.cpp \
    bright_sve.cpp conv_sve.cpp curv_sve.cpp gauss_errc_sve.cpp mrqcof_sve.cpp

# Define the object files
OBJ = $(SRC:%.cpp=$(BUILD_DIR)/%.o)

# Default target to build the executable
all: $(BUILD_DIR) $(TARGET)

# Rule to create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to build the static library
$(TARGET): $(OBJ) $(BUILD_DIR)/libstdc++.a $(BOINC_LIB_DIR)/libboinc.a $(BOINC_API_DIR)/libboinc_api.a $(PS_COMMON_LIB_DIR)/libps_common.a
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(BUILD_DIR)/libstdc++.a -pthread \
	$(PS_COMMON_LIB_DIR)/libps_common.a \
	$(BOINC_API_DIR)/libboinc_api.a \
	$(BOINC_LIB_DIR)/libboinc.a \
	-lm

# Rule to build object files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to create symbolic link for libstdc++.a
$(BUILD_DIR)/libstdc++.a:
	@mkdir -p $(BUILD_DIR)
	ln -s `g++ -print-file-name=libstdc++.a` $(BUILD_DIR)/libstdc++.a

# Clean rule to remove compiled files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

NVCC = nvcc
GCC_COMPILER = /usr/bin/gcc-9
NVCC_FLAGS = -O3 -std=c++11 -Xcompiler -fopenmp
CUDA_ARCH = -arch=sm_61
LIBS = -lstdc++

TARGET = prog
SOURCES = main.cu

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -ccbin=$(GCC_COMPILER) $(LIBS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)
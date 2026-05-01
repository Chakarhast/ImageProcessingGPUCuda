NVCC = nvcc
TARGET = pipeline

all:
	nvcc -std=c++17 src/main.cu src/streams.cu -o pipeline

clean:
	rm -f $(TARGET)
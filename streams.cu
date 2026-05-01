#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
namespace fs = std::filesystem;

// I am declaring existing kernels (defined in main.cu)
__global__ void rgb_to_grayscale(unsigned char*, unsigned char*, int, int, int);
__global__ void gaussian_blur(unsigned char*, unsigned char*, int, int);

extern void checkCuda(cudaError_t result);

// I am processing images in parallel using CUDA streams
float process_images_streams(const string& input_dir, const string& output_dir) {
    vector<string> images;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        images.push_back(entry.path().string());
    }

    int n = images.size();
    vector<cudaStream_t> streams(n);

    // I am creating streams
    for (int i = 0; i < n; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < n; i++) {
        int width, height, channels;

        unsigned char* h_input = stbi_load(images[i].c_str(), &width, &height, &channels, 3);
        if (!h_input) continue;

        int img_size = width * height * 3;
        int gray_size = width * height;

        unsigned char *d_input, *d_gray, *d_blur;
        checkCuda(cudaMalloc(&d_input, img_size));
        checkCuda(cudaMalloc(&d_gray, gray_size));
        checkCuda(cudaMalloc(&d_blur, gray_size));

        // I am copying asynchronously
        checkCuda(cudaMemcpyAsync(d_input, h_input, img_size, cudaMemcpyHostToDevice, streams[i]));

        int threads1D = 256;
        int blocks1D = (width * height + threads1D - 1) / threads1D;

        dim3 threads2D(16,16);
        dim3 blocks2D((width + 15)/16, (height + 15)/16);

        // I am launching kernels in stream
        rgb_to_grayscale<<<blocks1D, threads1D, 0, streams[i]>>>(d_input, d_gray, width, height, 3);
        gaussian_blur<<<blocks2D, threads2D, 0, streams[i]>>>(d_gray, d_blur, width, height);

        string filename = fs::path(images[i]).filename().string();
        string output_path = output_dir + "/" + filename;

        unsigned char* h_output = new unsigned char[gray_size];

        // I am copying back asynchronously
        checkCuda(cudaMemcpyAsync(h_output, d_blur, gray_size, cudaMemcpyDeviceToHost, streams[i]));

        // I am synchronizing per stream before writing
        cudaStreamSynchronize(streams[i]);

        stbi_write_png(output_path.c_str(), width, height, 1, h_output, width);

        cudaFree(d_input);
        cudaFree(d_gray);
        cudaFree(d_blur);
        stbi_image_free(h_input);
        delete[] h_output;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);

    // I am destroying streams
    for (int i = 0; i < n; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return time_ms;
}
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
namespace fs = std::filesystem;

float process_images_streams(const std::string& input_dir, const std::string& output_dir);

// I am implementing CPU grayscale for comparison
void cpu_grayscale(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int total = width * height;
    for (int i = 0; i < total; i++) {
        int idx = i * channels;

        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// I am defining CUDA kernel for grayscale conversion
__global__ void rgb_to_grayscale(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int i = idx * channels;

        unsigned char r = input[i];
        unsigned char g = input[i + 1];
        unsigned char b = input[i + 2];

        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// I am checking CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

// I am defining a 3x3 Gaussian blur kernel for single-channel images
__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int kernel[3][3] = {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };

    int sum = 0;
    int weight = 16;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);

            sum += input[ny * width + nx] * kernel[ky + 1][kx + 1];
        }
    }

    output[y * width + x] = sum / weight;
}

// I am implementing CPU blur for single-channel images
void cpu_blur(unsigned char* input, unsigned char* output, int width, int height) {
    int kernel[3][3] = {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };
    int weight = 16;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = min(max(x + kx, 0), width - 1);
                    int ny = min(max(y + ky, 0), height - 1);

                    sum += input[ny * width + nx] * kernel[ky + 1][kx + 1];
                }
            }

            output[y * width + x] = sum / weight;
        }
    }
}

void benchmark_image(const string& input_path, ofstream& log_file) {
    int width, height, channels;

    unsigned char* h_input = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
    if (!h_input) return;

    int img_size = width * height * 3;
    int gray_size = width * height;

    // host buffers
    unsigned char *h_gray = new unsigned char[gray_size];
    unsigned char *h_blur = new unsigned char[gray_size];
    unsigned char *h_tmp = new unsigned char[gray_size];

    // device buffers
    unsigned char *d_input, *d_gray, *d_blur;
    checkCuda(cudaMalloc(&d_input, img_size));
    checkCuda(cudaMalloc(&d_gray, gray_size));
    checkCuda(cudaMalloc(&d_blur, gray_size));

    checkCuda(cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice));

    int threads1D = 256;
    int blocks1D = (width * height + threads1D - 1) / threads1D;

    dim3 threads2D(16,16);
    dim3 blocks2D((width + 15)/16, (height + 15)/16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float g_gray=0, g_blur=0, g_both=0;
    float c_gray=0, c_blur=0, c_both=0;

    // ---------------- GRAYSCALE ----------------
    auto c1 = chrono::high_resolution_clock::now();
    cpu_grayscale(h_input, h_gray, width, height, 3);
    auto c2 = chrono::high_resolution_clock::now();
    c_gray = chrono::duration<float, milli>(c2-c1).count();

    cudaEventRecord(start);
    rgb_to_grayscale<<<blocks1D, threads1D>>>(d_input, d_gray, width, height, 3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_gray, start, stop);

    // ---------------- BLUR ONLY ----------------
    cpu_grayscale(h_input, h_tmp, width, height, 3); // convert first

    auto c3 = chrono::high_resolution_clock::now();
    cpu_blur(h_tmp, h_blur, width, height);
    auto c4 = chrono::high_resolution_clock::now();
    c_blur = chrono::duration<float, milli>(c4-c3).count();

    checkCuda(cudaMemcpy(d_gray, h_tmp, gray_size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    gaussian_blur<<<blocks2D, threads2D>>>(d_gray, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_blur, start, stop);

    // ---------------- BOTH ----------------
    auto c5 = chrono::high_resolution_clock::now();
    cpu_grayscale(h_input, h_tmp, width, height, 3);
    cpu_blur(h_tmp, h_blur, width, height);
    auto c6 = chrono::high_resolution_clock::now();
    c_both = chrono::duration<float, milli>(c6-c5).count();

    cudaEventRecord(start);
    rgb_to_grayscale<<<blocks1D, threads1D>>>(d_input, d_gray, width, height, 3);
    gaussian_blur<<<blocks2D, threads2D>>>(d_gray, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_both, start, stop);

    string name = fs::path(input_path).filename().string();

    log_file << name << ","
             << c_gray << "," << g_gray << ","
             << c_blur << "," << g_blur << ","
             << c_both << "," << g_both << "\n";

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);

    delete[] h_gray;
    delete[] h_blur;
    delete[] h_tmp;
    stbi_image_free(h_input);
}

// I am processing a single image
void process_image(const string& input_path, const string& output_path) {
    int width, height, channels;

    // I am loading image
    unsigned char* h_input = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
    if (!h_input) {
        cerr << "Failed to load image: " << input_path << endl;
        return;
    }

    int img_size = width * height * 3;
    int gray_size = width * height;

    // I am allocating host buffers
    unsigned char* h_gray = new unsigned char[gray_size];
    unsigned char* h_blur = new unsigned char[gray_size];
    unsigned char* h_cpu = new unsigned char[gray_size];

    // I am timing CPU execution
    auto cpu_start = chrono::high_resolution_clock::now();
    cpu_grayscale(h_input, h_cpu, width, height, 3);
    auto cpu_end = chrono::high_resolution_clock::now();
    float cpu_time = chrono::duration<float, milli>(cpu_end - cpu_start).count();

    // I am allocating device memory
    unsigned char *d_input, *d_gray, *d_blur;
    checkCuda(cudaMalloc(&d_input, img_size));
    checkCuda(cudaMalloc(&d_gray, gray_size));
    checkCuda(cudaMalloc(&d_blur, gray_size));

    // I am copying input to GPU
    checkCuda(cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice));

    // I am setting up execution configs
    int threads1D = 256;
    int blocks1D = (width * height + threads1D - 1) / threads1D;

    dim3 threads2D(16,16);
    dim3 blocks2D((width + 15)/16, (height + 15)/16);

    // I am creating CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // I am launching grayscale kernel
    rgb_to_grayscale<<<blocks1D, threads1D>>>(d_input, d_gray, width, height, 3);

    // I am launching blur kernel
    gaussian_blur<<<blocks2D, threads2D>>>(d_gray, d_blur, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // I am copying result back
    checkCuda(cudaMemcpy(h_blur, d_blur, gray_size, cudaMemcpyDeviceToHost));

    // I am saving output
    stbi_write_png(output_path.c_str(), width, height, 1, h_blur, width);

    cout << "Processed: " << input_path
         << " | CPU(ms): " << cpu_time
         << " | GPU(ms): " << gpu_time << endl;

    // I am appending to CSV
    static ofstream log_file("results.csv", ios::app);
    log_file << fs::path(input_path).filename().string()
             << "," << cpu_time << "," << gpu_time << "\n";

    // I am freeing memory
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);

    stbi_image_free(h_input);
    delete[] h_gray;
    delete[] h_blur;
    delete[] h_cpu;
}



int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: ./pipeline <input_dir> <output_dir>" << endl;
        return 1;
    }

    string input_dir = argv[1];
    string output_dir = argv[2];

    fs::create_directories(output_dir);

    // ---------- (OPTIONAL) Existing per-image pipeline ----------
    ofstream log_file_basic("results.csv");
    log_file_basic << "image,cpu_ms,gpu_ms\n";
    log_file_basic.close();

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        string input_path = entry.path().string();
        string filename = entry.path().filename().string();
        string output_path = output_dir + "/" + filename;

        process_image(input_path, output_path);
    }

    // ---------- NEW: Detailed benchmarking ----------
    ofstream log_file("final_results.csv");
    log_file << "image,gray_cpu,gray_gpu,blur_cpu,blur_gpu,both_cpu,both_gpu\n";

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        benchmark_image(entry.path().string(), log_file);
    }

    log_file.close();

    // ---------- Streams optimization ----------
    float optimized_time = process_images_streams(input_dir, output_dir);

    cout << "Optimized GPU Time (ms): " << optimized_time << endl;

    return 0;
}
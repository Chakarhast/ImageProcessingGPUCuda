# 🚀 GPU-Accelerated Image Processing Pipeline using CUDA

| Image 1 | Image 2 |
| :---: | :---: |
| ![Caption 1](13246512-fd26-4637-82f6-3573284e6c93.jpeg) | ![Caption 2](fd240882-9dbd-4f71-adae-5a321f350cd9.jpeg) |



## 📌 Project Motivation

Modern workloads in computer vision and signal processing involve applying the same operation across large datasets. These workloads are inherently **data-parallel**, making them ideal candidates for GPU acceleration.

This project explores:
- How image processing operations map to GPU parallelism
- The performance gap between CPU and GPU implementations
- How batching and CUDA streams improve throughput

The implementation focuses on building everything from scratch using **custom CUDA kernels**, rather than relying on high-level GPU libraries.

---

## 🎯 Objectives

1. Implement image processing operations using CUDA:
   - Grayscale conversion
   - Gaussian blur
   - Combined pipeline (grayscale → blur)

2. Compare:
   - CPU execution vs GPU execution
   - Sequential GPU vs stream-based GPU

3. Process a **large dataset (100 images)** to demonstrate scalability

4. Generate measurable artifacts:
   - CSV logs
   - Visual plots
   - Output images

---

## ⚙️ System Design

### 🔁 Processing Pipeline

```
Input Image (RGB)
        ↓
[CUDA Kernel] Grayscale Conversion
        ↓
[CUDA Kernel] Gaussian Blur
        ↓
Output Image (Processed)
```

---

### 🧵 Parallelization Strategy

- Each thread processes **one pixel**
- Total threads = `width × height`
- Threads are organized into:
  - 1D grid for grayscale
  - 2D grid for blur

---

### 💾 Memory Flow

```
Host (CPU)
   ↓ cudaMemcpy
Device (GPU)
   ↓ Kernel Execution
Device (GPU)
   ↓ cudaMemcpy
Host (CPU)
```

---

### ⚡ CUDA Streams (Optimization)

To improve throughput, a second implementation uses **CUDA streams**:

- Each image is processed in a separate stream
- Enables:
  - Overlapping memory transfers
  - Concurrent kernel execution
- Reduces idle GPU time

---

## 📂 Project Structure

```
.
├── src/
│   ├── main.cu              # Core pipeline + benchmarking logic
│   ├── streams.cu           # Stream-based parallel execution
│   ├── stb_image.h
│   ├── stb_image_write.h
│
├── data/
│   ├── input/               # 100 generated images
│   ├── output/              # processed outputs
│
├── plots/                   # visualization outputs
├── final_results.csv        # detailed benchmark results
├── results.csv              # basic pipeline results
├── analyze_results.py       # analysis + plotting
├── Makefile
├── run.sh
└── README.md
```

---

## 🛠️ Execution Instructions (Google Colab)

This project requires an NVIDIA GPU. Recommended environment: **Google Colab (Tesla T4)**.

---

### Step 1: Enable GPU

```
Runtime → Change runtime type → GPU
```

---

### Step 2: Clone Repository

```bash
!git clone <repo-link>
%cd <repo-name>
```

---

### Step 3: Generate Dataset

We use a controlled dataset to ensure fair benchmarking.

```bash
!mkdir -p data/input
!rm -f data/input/*
!for i in $(seq 1 100); do wget -q -O data/input/img_$i.jpg https://picsum.photos/256/256; done
```

Why this dataset:
- Fixed resolution (256×256)
- Removes variability due to image size
- Ensures consistent benchmarking

---

### Step 4: Build

```bash
!make clean
!make
```

---

### Step 5: Run

```bash
!./pipeline data/input data/output
```

---

### Step 6: Verify GPU

```bash
!nvidia-smi
```

---

### Step 7: Analyze Results

```bash
!python analyze_results.py
```

---

## 📊 Benchmarking Methodology

For each image, the following are measured:

| Operation | CPU Time | GPU Time |
|----------|--------|---------|
| Grayscale | ✔ | ✔ |
| Blur | ✔ | ✔ |
| Pipeline | ✔ | ✔ |

Output file:
```
final_results.csv
```

Format:
```
image,gray_cpu,gray_gpu,blur_cpu,blur_gpu,both_cpu,both_gpu
```

---

## 📈 Performance Results

### Key Observations

1. **GPU outperforms CPU consistently**
   - Due to parallel execution across pixels

2. **Blur achieves higher speedup**
   - More compute-intensive → better GPU utilization

3. **Pipeline benefits compound**
   - Multiple kernels amplify performance gains

4. **Low variance across images**
   - Due to controlled dataset size

---

## 📊 Visualization

Generated plots include:

- Average CPU vs GPU time
- Speedup comparison
- CPU vs GPU scatter plot
- Speedup distribution
- Per-image variation

These are stored in:

```
plots/
```

---

## ⚡ Streams Optimization Results

A second implementation uses CUDA streams.

### Improvements:
- Overlapping computation + memory transfer
- Increased GPU utilization
- Reduced total execution time for batch

---

## 🧠 Technical Insights

- GPU excels in **SIMD-style workloads**
- Memory transfer overhead is non-trivial
- Kernel launch overhead becomes negligible at scale
- Batch size significantly affects performance
- Streams enable **task-level parallelism**, not just data-level

---

## 🚧 Challenges Encountered

- Handling multiple CUDA files without symbol conflicts
- Managing host/device memory correctly
- Ensuring fair CPU vs GPU comparison
- Debugging kernel launch configurations
- Maintaining reproducibility across runs

---

## 🔮 Future Work

- Multi-GPU scaling
- Larger resolution images
- Additional filters (edge detection, sharpening)
- Use of shared memory for optimization
- Integration with CUDA libraries (NPP)

---

## 🎥 Demonstration

A 5–10 minute video includes:
- Architecture walkthrough
- Code explanation
- Performance analysis
- Future improvements

(Insert link here)

---

## 📌 Conclusion

This project demonstrates that GPU acceleration provides **significant performance improvements** for data-parallel workloads. By combining kernel-based parallelism with stream-based execution, the system achieves both **efficiency and scalability**, making it suitable for real-world high-throughput processing systems.

---

## 🔗 References

- NVIDIA CUDA Programming Guide  
- STB Image Library  
- Picsum Photos Dataset  
- CUDA Runtime Documentation  

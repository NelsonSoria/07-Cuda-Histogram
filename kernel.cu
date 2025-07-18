#include <cstdint>
#include <cuda_runtime.h>

__global__ void histogram_kernel(const uint8_t* img, int img_size, int* hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < img_size) {
        atomicAdd(&hist[img[idx]], 1);
    }
}

void histogramCUDA(const uint8_t* img, int img_size, int* hist_host) {
    uint8_t* d_img;
    int* d_hist;
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    int blockSize = 256;
    int numBlocks = (img_size + blockSize - 1) / blockSize;
    histogram_kernel<<<numBlocks, blockSize>>>(d_img, img_size, d_hist);

    cudaMemcpy(hist_host, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_hist);
}
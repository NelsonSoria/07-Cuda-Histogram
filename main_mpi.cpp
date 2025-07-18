#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <mpi.h>

// MPI histogram function
void histogramMPI(const uint8_t* img, int img_size, int* hist, int rank, int size) {
    std::vector<int> local_hist(256, 0);
    int chunk = img_size / size;
    int start = rank * chunk;
    int end = (rank == size-1) ? img_size : start + chunk;
    for (int i = start; i < end; ++i)
        local_hist[img[i]]++;
    MPI_Reduce(local_hist.data(), hist, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string input_image = "image.png";
    std::string output_hist_img = "histograma_mpi.png";

    int width, height, channels;
    uint8_t* gray_pixels = stbi_load(input_image.c_str(), &width, &height, &channels, STBI_grey);
    if (!gray_pixels) {
        if (rank == 0)
            std::cerr << "No se pudo leer la imagen.\n";
        MPI_Finalize();
        return 1;
    }
    int img_size = width * height;
    std::vector<int> hist(256, 0);

    histogramMPI(gray_pixels, img_size, hist.data(), rank, size);

    if (rank == 0) {
        // --- Crear imagen del histograma ---
        const int hist_w = 512;
        const int hist_h = 400;
        std::vector<uint8_t> hist_img(hist_w * hist_h, 255);

        int max_val = *std::max_element(hist.begin(), hist.end());
        for (int x = 0; x < 256; ++x) {
            int h = (int)((hist[x] / (float)max_val) * (hist_h - 1));
            for (int y = 0; y < h; ++y) {
                int px = x * 2;
                int py = hist_h - 1 - y;
                if (px < hist_w && py < hist_h)
                    hist_img[py * hist_w + px] = 0;
                if (px + 1 < hist_w && py < hist_h)
                    hist_img[py * hist_w + px + 1] = 0;
            }
        }
        stbi_write_png(output_hist_img.c_str(), hist_w, hist_h, 1, hist_img.data(), hist_w);
        std::cout << "Histograma guardado como imagen en " << output_hist_img << "\n";
    }
    stbi_image_free(gray_pixels);
    MPI_Finalize();
    return 0;
}


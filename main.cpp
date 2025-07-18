#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h> // AVX2

// CUDA
void histogramCUDA(const uint8_t* img, int img_size, int* hist_host);

// Serial
void histogramSerial(const uint8_t* img, int img_size, int* hist) {
    std::fill(hist, hist + 256, 0);
    for (int i = 0; i < img_size; ++i)
        hist[img[i]]++;
}

// SIMD (SSE2)
void histogramSIMD(const uint8_t* img, int img_size, int* hist) {
    std::fill(hist, hist + 256, 0);
    alignas(32) int local_hist[256] = {0};
    int i = 0;

    // Procesa 32 píxeles a la vez
    for (; i <= img_size - 32; i += 32) {
        __m256i pixels = _mm256_loadu_si256((const __m256i*)(img + i));
        alignas(32) uint8_t vals[32];
        _mm256_store_si256((__m256i*)vals, pixels);
        for (int j = 0; j < 32; ++j)
            local_hist[vals[j]]++;
    }
    // Resto escalar
    for (; i < img_size; ++i)
        local_hist[img[i]]++;

    for (int k = 0; k < 256; ++k)
        hist[k] = local_hist[k];
}

// OpenMP
void histogramOpenMP(const uint8_t* img, int img_size, int* hist) {
    int local_hist[256] = {0};
    #pragma omp parallel
    {
        int priv_hist[256] = {0};
        #pragma omp for nowait
        for (int i = 0; i < img_size; ++i)
            priv_hist[img[i]]++;
        #pragma omp critical
        for (int i = 0; i < 256; ++i)
            local_hist[i] += priv_hist[i];
    }
    std::copy(local_hist, local_hist + 256, hist);
}

int main(int argc, char* argv[]) {
    while (true) {
        std::cout << "Seleccione el modo de ejecucion:\n";
        std::cout << "0. Salir\n";
        std::cout << "1. Serial\n";
        std::cout << "2. SIMD\n";
        std::cout << "3. OpenMP\n";
        std::cout << "4. CUDA\n";
        std::cout << "Ingrese el numero de la opcion: ";
        int opcion = 0;
        std::cin >> opcion;

        if (opcion == 0) {
            std::cout << "Saliendo...\n";
            break;
        }

        std::string mode;
        switch (opcion) {
            case 1: mode = "serial"; break;
            case 2: mode = "simd"; break;
            case 3: mode = "openmp"; break;
            case 4: mode = "cuda"; break;
            default:
                std::cerr << "Opcion no valida.\n";
                continue;
        }

        std::string input_image = "image2.png";
        std::string output_hist_img = "histograma_" + mode + ".png";

        int width, height, channels;
        uint8_t* gray_pixels = stbi_load(input_image.c_str(), &width, &height, &channels, STBI_grey);
        if (!gray_pixels) {
            std::cerr << "No se pudo leer la imagen.\n";
            continue;
        }
        int img_size = width * height;
        std::vector<int> hist(256, 0);

        if (mode == "serial") {
            histogramSerial(gray_pixels, img_size, hist.data());
        } else if (mode == "simd") {
            histogramSIMD(gray_pixels, img_size, hist.data());
        } else if (mode == "openmp") {
            histogramOpenMP(gray_pixels, img_size, hist.data());
        } else if (mode == "cuda") {
            histogramCUDA(gray_pixels, img_size, hist.data());
        }
        stbi_image_free(gray_pixels);

        // --- Crear imagen del histograma ---
        const int margin = 20;
        const int hist_w = 512;
        const int hist_h = 400;
        std::vector<uint8_t> hist_img(hist_w * hist_h, 255);

        int max_val = *std::max_element(hist.begin(), hist.end());
        for (int x = 0; x < 256; ++x) {
            int h = (int)((hist[x] / (float)max_val) * (hist_h - 1 - margin));
            for (int y = 0; y < h; ++y) {
                int px = x * 2;
                int py = hist_h - 1 - y - margin;
                if (px < hist_w && py < hist_h)
                    hist_img[py * hist_w + px] = 0;
                if (px + 1 < hist_w && py < hist_h)
                    hist_img[py * hist_w + px + 1] = 0;
            }
        }
        // Eje X
        for (int x = 0; x < hist_w; ++x)
            hist_img[(hist_h - margin - 1) * hist_w + x] = 100;

        stbi_write_png(output_hist_img.c_str(), hist_w, hist_h, 1, hist_img.data(), hist_w);
        std::cout << "Histograma guardado como imagen en " << output_hist_img << "\n\n";
    }
    return 0;
}
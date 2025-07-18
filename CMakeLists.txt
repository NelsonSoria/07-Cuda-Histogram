cmake_minimum_required(VERSION 3.16)

project(CudaHistogram)

SET(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)

find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS Graphics Window System CONFIG REQUIRED)

# --- Configuración para MPI ---
option(USE_MPI "Enable MPI support" ON)
if(USE_MPI)
    set(MPI_ROOT "C:/Program Files (x86)/Intel/oneAPI/mpi/latest")
    set(MPI_INCLUDE "${MPI_ROOT}/include")
    set(MPI_LIB     "${MPI_ROOT}/lib")
    include_directories(${MPI_INCLUDE})
endif()

add_executable(cuda_histogram 
    main.cpp
    kernel.cu
)

add_executable(mpi_histogram
    main_mpi.cpp
)

target_include_directories(cuda_histogram PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(mpi_histogram PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(cuda_histogram PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_STANDARD 17
)

target_link_libraries(cuda_histogram PRIVATE
    fmt::fmt
    SFML::Graphics SFML::Window SFML::System
)

target_link_libraries(mpi_histogram PRIVATE
    fmt::fmt
    SFML::Graphics SFML::Window SFML::System
)

if(USE_MPI)
    target_compile_definitions(mpi_histogram PRIVATE USE_MPI)
    target_link_libraries(mpi_histogram PRIVATE
        ${MPI_LIB}/impi.lib
        #${MPI_LIB}/impicxx.lib
    )
endif()

if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2)
endif()

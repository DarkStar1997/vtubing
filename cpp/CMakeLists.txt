cmake_minimum_required(VERSION 3.20)
project(vtuber_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
find_package(SDL3 REQUIRED)

include(FetchContent)
set(FETCH_CONTENT_QUIET OFF)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
    GIT_SHALLOW TRUE)

FetchContent_Declare(bs_thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG v5.0.0
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glm bs_thread_pool)

# cgltf + stb are single-header libraries — download directly
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(DOWNLOAD
    https://raw.githubusercontent.com/jkuhlmann/cgltf/v1.14/cgltf.h
    ${THIRD_PARTY_DIR}/cgltf.h
    STATUS cgltf_status)
file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    ${THIRD_PARTY_DIR}/stb_image.h
    STATUS stbi_status)
file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    ${THIRD_PARTY_DIR}/stb_image_write.h
    STATUS stbw_status)

# --- Target ---
add_executable(vtuber_cpu
    src/main.cpp
    src/vrm_loader.cpp
    src/rasterizer.cpp)

target_include_directories(vtuber_cpu PRIVATE
    src
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(vtuber_cpu PRIVATE SDL3::SDL3 glm::glm)

# AVX2 + FMA support
target_compile_options(vtuber_cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mbmi2>)

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

# --- OpenCV (system) ---
find_package(OpenCV REQUIRED)

# --- ONNX Runtime (pre-built) ---
file(GLOB ORT_DIR "${THIRD_PARTY_DIR}/onnxruntime-linux-x64-*")
if(NOT ORT_DIR)
    message(FATAL_ERROR "ONNX Runtime not found in ${THIRD_PARTY_DIR}. Download onnxruntime-linux-x64-1.28.0.tgz and extract.")
endif()
message(STATUS "ONNX Runtime: ${ORT_DIR}")

# --- Shared rendering sources (used by both targets) ---
set(RENDER_SOURCES
    src/vrm_loader.cpp
    src/rasterizer.cpp
    src/rasterizer_tri.cpp)

# --- Benchmark / offline renderer target ---
add_executable(vtuber_cpu
    src/main.cpp
    ${RENDER_SOURCES})

target_include_directories(vtuber_cpu PRIVATE
    src
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(vtuber_cpu PRIVATE SDL3::SDL3 glm::glm)

# AVX2 + FMA support
target_compile_options(vtuber_cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mbmi2>)

# Fast-math only on rasterizeTri (separate file to avoid breaking vertex transform)
set_source_files_properties(src/rasterizer_tri.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")

# --- Live tracking target (webcam + ONNX inference + real-time render) ---
add_executable(vtuber_live
    src/main_live.cpp
    src/webcam.cpp
    src/onnx_session.cpp
    src/face_tracker.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(vtuber_live PRIVATE
    src
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include
    ${ORT_DIR}/include)

target_link_libraries(vtuber_live PRIVATE
    SDL3::SDL3 glm::glm
    ${OpenCV_LIBS}
    onnxruntime)

target_link_directories(vtuber_live PRIVATE ${ORT_DIR}/lib)

# RPATH so the binary finds libonnxruntime.so at runtime
set_target_properties(vtuber_live PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN")

target_compile_options(vtuber_live PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mbmi2>)

set_source_files_properties(src/rasterizer_tri.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")

# --- Test tracker target (static image inference) ---
add_executable(test_tracker
    src/test_tracker.cpp
    src/onnx_session.cpp
    src/face_tracker.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(test_tracker PRIVATE
    src
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include
    ${ORT_DIR}/include)

target_link_libraries(test_tracker PRIVATE
    glm::glm
    ${OpenCV_LIBS}
    onnxruntime)

target_link_directories(test_tracker PRIVATE ${ORT_DIR}/lib)

set_target_properties(test_tracker PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN")

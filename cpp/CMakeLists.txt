cmake_minimum_required(VERSION 3.20)
project(vtuber_cpu LANGUAGES CXX C)

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

# --- MediaPipe C API (pre-built shared library) ---
if(WIN32)
    set(MP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.dll)
else()
    set(MP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.so)
endif()

if(MSVC)
    # cgltf/stb use fopen and other POSIX-named CRT functions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()


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
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mbmi2>)

# --- Live tracking target (webcam + MediaPipe inference + real-time render) ---
set(VTUBER_LIVE_SOURCES
    src/main_live.cpp
    src/webcam.cpp
    src/face_tracker.cpp
    src/pose_tracker.cpp
    src/hand_tracker.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})
if(NOT WIN32)
    # EGL symbol stubs force MediaPipe onto pure-CPU path on Linux (compiled
    # in with -rdynamic so they override libEGL at link time).
    list(APPEND VTUBER_LIVE_SOURCES src/egl_stub.c)
endif()

add_executable(vtuber_live ${VTUBER_LIVE_SOURCES})

target_include_directories(vtuber_live PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

if(NOT WIN32)
    target_link_options(vtuber_live PRIVATE -rdynamic)
endif()

target_link_libraries(vtuber_live PRIVATE
    SDL3::SDL3 glm::glm
    ${MP_LIB})

if(UNIX)
    set_target_properties(vtuber_live PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        INSTALL_RPATH "$ORIGIN")
endif()

target_compile_options(vtuber_live PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mbmi2>)

# --- Test tracker target (static image inference) ---
add_executable(test_tracker
    src/test_tracker.cpp
    src/face_tracker.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(test_tracker PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(test_tracker PRIVATE
    glm::glm
    ${MP_LIB})

if(UNIX)
    set_target_properties(test_tracker PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        INSTALL_RPATH "$ORIGIN")
endif()

# --- Windows: copy runtime DLLs next to the executables ---
if(WIN32)
    foreach(tgt vtuber_live test_tracker)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${MP_LIB} $<TARGET_FILE_DIR:${tgt}>)
    endforeach()
    foreach(tgt vtuber_live vtuber_cpu)
        if(TARGET SDL3::SDL3-shared)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:SDL3::SDL3-shared> $<TARGET_FILE_DIR:${tgt}>)
        endif()
    endforeach()
endif()

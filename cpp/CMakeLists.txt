cmake_minimum_required(VERSION 3.20)
project(vtuber_cpu LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the MediaPipe wheel dylib's minimum (macOS 14) so release builds run
# on older systems than the build machine's SDK.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0 CACHE STRING "Minimum macOS version" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
include(FetchContent)
set(FETCH_CONTENT_QUIET OFF)

# SDL3: prefer a system/package install (apt, pacman, vcpkg, or
# CMAKE_PREFIX_PATH) when available; otherwise fetch the pinned release
# and build it from source as part of this project.
find_package(SDL3 QUIET)
if(NOT SDL3_FOUND)
    message(STATUS "SDL3 not found - fetching release-3.4.16 and building from source")
    # Keep the vendored build lean: shared lib only, no test scaffolding.
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-3.4.16
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
endif()

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
    # MSVC's linker cannot link a DLL directly (LNK1107): link against an
    # import library and copy the DLL next to the executables post-build.
    set(MP_DLL ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.dll)
    set(MP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.lib)
    set(MP_VENV_PATH .venv/Lib/site-packages/mediapipe/tasks/c/libmediapipe.dll)
elseif(APPLE)
    set(MP_DLL)
    set(MP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.dylib)
    set(MP_VENV_PATH .venv/lib/python3.12/site-packages/mediapipe/tasks/c/libmediapipe.dylib)
else()
    set(MP_DLL)
    set(MP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe.so)
    set(MP_VENV_PATH .venv/lib/python3.12/site-packages/mediapipe/tasks/c/libmediapipe.so)
endif()

# Auto-copy from the Python venv if the library is missing.
if(WIN32)
    if(NOT EXISTS ${MP_DLL})
        set(MP_VENV_CANDIDATE ${CMAKE_SOURCE_DIR}/../${MP_VENV_PATH})
        if(MP_VENV_CANDIDATE AND EXISTS ${MP_VENV_CANDIDATE})
            message(STATUS "Copying libmediapipe from venv: ${MP_VENV_CANDIDATE}")
            file(COPY ${MP_VENV_CANDIDATE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
        else()
            message(FATAL_ERROR
                "libmediapipe not found at ${MP_DLL}\n"
                "Run 'uv sync' in the repo root first (install uv from\n"
                "https://docs.astral.sh/uv/ if needed). This creates .venv/ with\n"
                "the mediapipe wheel; re-run CMake and the DLL will be copied\n"
                "automatically. Manual copy:\n"
                "  copy .venv\\Lib\\site-packages\\mediapipe\\tasks\\c\\libmediapipe.dll to cpp\\lib\\")
        endif()
    endif()

    # The MediaPipe wheel ships no import library: generate one from the
    # DLL's exported symbols. Requires dumpbin/lib on PATH (VS dev prompt).
    if(NOT EXISTS ${MP_LIB})
        find_program(DUMPBIN_EXECUTABLE dumpbin)
        find_program(LIB_EXECUTABLE lib)
        if(NOT DUMPBIN_EXECUTABLE OR NOT LIB_EXECUTABLE)
            message(FATAL_ERROR
                "Cannot generate ${MP_LIB}: dumpbin/lib not found on PATH.\n"
                "Configure from an x64 Native Tools Command Prompt for VS\n"
                "(or a shell that has run vcvarsall.bat x64), or provide\n"
                "libmediapipe.lib manually in cpp/lib/.")
        endif()
        message(STATUS "Generating import library: ${MP_LIB}")
        execute_process(
            COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                    -File ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_mediapipe_implib.ps1
                    ${MP_DLL} ${MP_LIB}
            RESULT_VARIABLE mp_implib_result)
        if(NOT mp_implib_result EQUAL 0)
            message(FATAL_ERROR "Failed to generate import library for ${MP_DLL}")
        endif()
    endif()
else()
    if(NOT EXISTS ${MP_LIB})
        set(MP_VENV_CANDIDATE ${CMAKE_SOURCE_DIR}/../${MP_VENV_PATH})
        if(NOT EXISTS ${MP_VENV_CANDIDATE})
            if(APPLE)
                file(GLOB MP_VENV_GLOB
                    ${CMAKE_SOURCE_DIR}/../.venv/lib/python*/site-packages/mediapipe/tasks/c/libmediapipe.dylib)
            else()
                file(GLOB MP_VENV_GLOB
                    ${CMAKE_SOURCE_DIR}/../.venv/lib/python*/site-packages/mediapipe/tasks/c/libmediapipe.so)
            endif()
            list(GET MP_VENV_GLOB 0 MP_VENV_CANDIDATE)
        endif()
        if(MP_VENV_CANDIDATE AND EXISTS ${MP_VENV_CANDIDATE})
            message(STATUS "Copying libmediapipe from venv: ${MP_VENV_CANDIDATE}")
            file(COPY ${MP_VENV_CANDIDATE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
        else()
            message(FATAL_ERROR
                "libmediapipe not found at ${MP_LIB}\n"
                "Run 'uv sync' in the repo root first (install uv from\n"
                "https://docs.astral.sh/uv/ if needed). This creates .venv/ with\n"
                "the mediapipe wheel; re-run CMake and the library will be copied\n"
                "automatically. Manual copy:\n"
                "  cp .venv/lib/python3.*/site-packages/mediapipe/tasks/c/libmediapipe.* cpp/lib/")
        endif()
    endif()
    if(APPLE)
        # The macOS wheel's dylib carries install name "@rpath/libmediapipe_source.so";
        # executables linked against it record that name as their dependency.
        # Provide the name as a symlink so rpath resolution finds the library.
        if(EXISTS ${MP_LIB} AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe_source.so)
            file(CREATE_LINK libmediapipe.dylib
                ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmediapipe_source.so SYMBOLIC)
        endif()
    endif()
endif()

if(MSVC)
    # cgltf/stb use fopen and other POSIX-named CRT functions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# AVX2 + FMA support (x86_64 only; Apple Silicon / other archs build portable code)
set(VT_SIMD_OPTS)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    if(MSVC)
        list(APPEND VT_SIMD_OPTS /arch:AVX2)
    else()
        list(APPEND VT_SIMD_OPTS -mavx2 -mfma -mbmi2)
    endif()
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

# AVX2 + FMA support (x86_64 only, see VT_SIMD_OPTS above)
target_compile_options(vtuber_cpu PRIVATE ${VT_SIMD_OPTS})

# --- Live tracking target (webcam + MediaPipe inference + real-time render) ---
set(VTUBER_LIVE_SOURCES
    src/main_live.cpp
    src/webcam.cpp
    src/face_tracker.cpp
    src/pose_tracker.cpp
    src/hand_tracker.cpp
    src/rig_solver.cpp
    src/springbone.cpp
    ${RENDER_SOURCES})
if(UNIX AND NOT APPLE)
    # EGL symbol stubs force MediaPipe onto pure-CPU path on Linux (compiled
    # in with -rdynamic so they override libEGL at link time). macOS has no
    # EGL and the frameworks-based wheel does not need this.
    list(APPEND VTUBER_LIVE_SOURCES src/egl_stub.c)
endif()

add_executable(vtuber_live ${VTUBER_LIVE_SOURCES})

target_include_directories(vtuber_live PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

if(UNIX AND NOT APPLE)
    target_link_options(vtuber_live PRIVATE -rdynamic)
endif()

target_link_libraries(vtuber_live PRIVATE
    SDL3::SDL3 glm::glm
    ${MP_LIB})

if(UNIX)
    set_target_properties(vtuber_live PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        INSTALL_RPATH "$ORIGIN")
    if(APPLE)
        # ld64 resolves "@loader_path"; $ORIGIN support varies by linker version
        set_target_properties(vtuber_live PROPERTIES INSTALL_RPATH "@loader_path")
    endif()
endif()

# Windows: build vtuber_live as a GUI-subsystem app so no console window
# opens on double-click. mainCRTStartup keeps plain main() as the entry
# point; at runtime the app re-attaches to the parent console when it was
# launched from a terminal (see main_live.cpp), so --verbose still works.
if(WIN32)
    set_target_properties(vtuber_live PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

target_compile_options(vtuber_live PRIVATE ${VT_SIMD_OPTS})

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
    if(APPLE)
        set_target_properties(test_tracker PROPERTIES INSTALL_RPATH "@loader_path")
    endif()
endif()

# --- Calibration regression test (synthetic face, no camera needed) ---
add_executable(test_calibration
    src/test_calibration.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(test_calibration PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(test_calibration PRIVATE
    glm::glm)

if(UNIX)
    set_target_properties(test_calibration PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        INSTALL_RPATH "$ORIGIN")
endif()

# --- Rig regression tests (synthetic faces, no camera/tracking needed) ---
# test_rig: ARKit→VRM expression mapping, blink combining, ARKit passthrough,
# gaze angle math, head gain/clamp knobs, 0.x preset rename.
add_executable(test_rig
    src/test_rig.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(test_rig PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(test_rig PRIVATE glm::glm)

# test_springbone: verlet chain falls under gravity, colliders push out,
# bone length is preserved.
add_executable(test_springbone
    src/test_springbone.cpp
    src/springbone.cpp)

target_include_directories(test_springbone PRIVATE
    src
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(test_springbone PRIVATE glm::glm)

# test_render: expression-driven render pixel diffs (port of Python
# test_rig.py): neutral vs aa/blink/gaze, transparent-background alpha.
# Needs a VRM file: default avatar, or first CLI argument.
add_executable(test_render
    src/test_render.cpp
    src/rig_solver.cpp
    ${RENDER_SOURCES})

target_include_directories(test_render PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include)

target_link_libraries(test_render PRIVATE glm::glm)

# test_vrm_loader: VRM 1.0 expressions / lookAt / springbone parsing against
# a checked-in fixture.
add_executable(test_vrm_loader
    src/test_vrm_loader.cpp
    src/vrm_loader.cpp)

target_include_directories(test_vrm_loader PRIVATE
    src
    ${THIRD_PARTY_DIR})

target_link_libraries(test_vrm_loader PRIVATE glm::glm)

enable_testing()
foreach(t test_calibration test_rig test_springbone)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
# test_render needs the avatar model from the repo's assets directory
add_test(NAME test_render COMMAND test_render
    ${CMAKE_CURRENT_SOURCE_DIR}/../assets/avatars/male_52blendshapes.vrm)
add_test(NAME test_vrm_loader COMMAND test_vrm_loader
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/vrm1_minimal.vrm)

# --- Windows: copy runtime DLLs next to the executables ---
if(WIN32)
    foreach(tgt vtuber_live test_tracker)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${MP_DLL} $<TARGET_FILE_DIR:${tgt}>)
    endforeach()
    foreach(tgt vtuber_live vtuber_cpu)
        if(TARGET SDL3::SDL3-shared)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:SDL3::SDL3-shared> $<TARGET_FILE_DIR:${tgt}>)
        endif()
    endforeach()
endif()

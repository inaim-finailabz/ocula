# Flutter Llama - CMake configuration for Android with llama.cpp
# Based on llama.cpp/examples/llama.android approach

cmake_minimum_required(VERSION 3.10.2)

project(flutter_llama)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Find required packages
find_library(log-lib log)
find_library(android-lib android)

# llama.cpp configuration
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party")

# Configure llama.cpp build
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Disable features not needed for mobile
set(GGML_USE_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_AMX OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
# Force Vulkan off — NDK Vulkan lacks vulkan.hpp C++ bindings needed by shader gen
set(vulkan-lib "" CACHE FILEPATH "" FORCE)

# ── OpenCL GPU acceleration (Adreno-optimized) ──
# Qualcomm Adreno GPUs perform significantly better with OpenCL than Vulkan.
# Most Android flagships (Nothing, OnePlus, Samsung US, older Pixels) use Adreno.
# The OpenCL library (libOpenCL.so) ships on Qualcomm devices at runtime.
# We provide Khronos headers from third_party/opencl-headers.

set(OPENCL_HEADERS_DIR "${THIRD_PARTY_DIR}/opencl-headers")
find_library(opencl-lib OpenCL)

if(opencl-lib AND EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)

    # Provide OpenCL package variables so find_package(OpenCL) succeeds
    set(OpenCL_FOUND TRUE CACHE BOOL "" FORCE)
    set(OpenCL_INCLUDE_DIRS "${OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
    set(OpenCL_LIBRARIES "${opencl-lib}" CACHE FILEPATH "" FORCE)

    # Create imported target for OpenCL
    if(NOT TARGET OpenCL::OpenCL)
        add_library(OpenCL::OpenCL SHARED IMPORTED)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "${opencl-lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_HEADERS_DIR}"
        )
    endif()

    message(STATUS "OpenCL GPU acceleration ENABLED (Adreno-optimized)")
    message(STATUS "  Headers: ${OPENCL_HEADERS_DIR}")
    message(STATUS "  Library: ${opencl-lib}")
else()
    set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
    if(NOT opencl-lib)
        message(STATUS "OpenCL disabled: libOpenCL not found (non-Qualcomm device?)")
    else()
        message(STATUS "OpenCL disabled: missing headers at ${OPENCL_HEADERS_DIR}")
    endif()
endif()

# ── Vulkan GPU acceleration (fallback for Mali/other GPUs) ──
# The NDK ships libvulkan.so + C headers, but NOT vulkan.hpp (C++ bindings).
# We supply vulkan.hpp from third_party/vulkan-hpp.
# glslc shader compiler comes from the NDK's shader-tools.

find_library(vulkan-lib vulkan)
set(VULKAN_HPP_DIR "${THIRD_PARTY_DIR}/vulkan-hpp")

if(vulkan-lib AND EXISTS "${VULKAN_HPP_DIR}/vulkan.hpp")
    # Tell llama.cpp to build Vulkan backend
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)

    # NDK sysroot already has vulkan/vulkan_core.h etc.
    # We need vulkan.hpp to sit next to them at <vulkan/vulkan.hpp>.
    # Create a wrapper include dir: third_party/vulkan-hpp is used as
    # an additional include with files at vulkan-hpp/vulkan.hpp, but
    # the #include expects <vulkan/vulkan.hpp>.  So we point at the
    # parent directory of vulkan-hpp.
    # Actually, the files live at third_party/vulkan-hpp/*.hpp and the
    # include is <vulkan/vulkan.hpp>, so we need a directory layout:
    #   <some_dir>/vulkan/vulkan.hpp
    # We'll create a symlink or just use the include path trick.

    # The NDK sysroot include dir already has a vulkan/ subdirectory with
    # the C headers.  We can overlay the C++ headers by adding vulkan-hpp's
    # parent as an include path.  But vulkan-hpp files are NOT in a vulkan/
    # subfolder.  Fix: we need them at third_party/vulkan-hpp/vulkan/*.hpp

    # Check if we already have the proper layout
    if(NOT EXISTS "${VULKAN_HPP_DIR}/vulkan")
        # Create proper vulkan/ subdirectory layout via symlink at configure time
        file(MAKE_DIRECTORY "${VULKAN_HPP_DIR}/vulkan")
        file(GLOB VK_HPP_FILES "${VULKAN_HPP_DIR}/*.hpp" "${VULKAN_HPP_DIR}/*.cppm")
        foreach(vkf ${VK_HPP_FILES})
            get_filename_component(vkfname "${vkf}" NAME)
            if(NOT EXISTS "${VULKAN_HPP_DIR}/vulkan/${vkfname}")
                file(CREATE_LINK "${vkf}" "${VULKAN_HPP_DIR}/vulkan/${vkfname}" SYMBOLIC)
            endif()
        endforeach()
    endif()

    # Provide Vulkan package variables so find_package(Vulkan) succeeds inside llama.cpp
    set(Vulkan_FOUND TRUE CACHE BOOL "" FORCE)
    set(Vulkan_INCLUDE_DIR "${VULKAN_HPP_DIR}" CACHE PATH "" FORCE)
    set(VULKAN_INCLUDE_DIR "${VULKAN_HPP_DIR}" CACHE PATH "" FORCE)
    set(Vulkan_LIBRARY "${vulkan-lib}" CACHE FILEPATH "" FORCE)
    set(VULKAN_LIBRARY "${vulkan-lib}" CACHE FILEPATH "" FORCE)

    # Create imported targets that llama.cpp's CMake expects
    if(NOT TARGET Vulkan::Vulkan)
        add_library(Vulkan::Vulkan SHARED IMPORTED)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            IMPORTED_LOCATION "${vulkan-lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${VULKAN_HPP_DIR}"
        )
    endif()

    # Find glslc in the NDK's shader-tools
    # The NDK bundles glslc at <ndk>/shader-tools/<host>/glslc
    get_filename_component(NDK_DIR "${CMAKE_ANDROID_NDK}" ABSOLUTE)
    find_program(Vulkan_GLSLC_EXECUTABLE
        NAMES glslc
        HINTS "${NDK_DIR}/shader-tools/darwin-x86_64"
              "${NDK_DIR}/shader-tools/linux-x86_64"
              "${NDK_DIR}/shader-tools/windows-x86_64"
        NO_DEFAULT_PATH
    )

    if(Vulkan_GLSLC_EXECUTABLE)
        set(Vulkan_glslc_FOUND TRUE CACHE BOOL "" FORCE)
        if(NOT TARGET Vulkan::glslc)
            add_executable(Vulkan::glslc IMPORTED)
            set_target_properties(Vulkan::glslc PROPERTIES
                IMPORTED_LOCATION "${Vulkan_GLSLC_EXECUTABLE}"
            )
        endif()
        message(STATUS "Vulkan GPU acceleration ENABLED")
        message(STATUS "  glslc: ${Vulkan_GLSLC_EXECUTABLE}")
        message(STATUS "  vulkan.hpp: ${VULKAN_HPP_DIR}")
    else()
        # No glslc — can't compile shaders, fall back to CPU
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
        message(STATUS "Vulkan disabled: glslc not found in NDK")
    endif()
else()
    message(STATUS "Vulkan disabled: missing libvulkan or vulkan-hpp headers")
    message(STATUS "  Using optimized CPU backend with ARM NEON")
endif()

# Summary
if(GGML_OPENCL AND GGML_VULKAN)
    message(STATUS "GPU: OpenCL (primary, Adreno) + Vulkan (fallback, Mali/other)")
elseif(GGML_OPENCL)
    message(STATUS "GPU: OpenCL only (Adreno-optimized)")
elseif(GGML_VULKAN)
    message(STATUS "GPU: Vulkan only")
else()
    message(STATUS "GPU: NONE — CPU-only with ARM NEON")
endif()
message(STATUS "Works on: All ARM64 Android devices")

# Add llama.cpp as subdirectory (it will build its own CMakeLists.txt)
add_subdirectory(${LLAMA_CPP_DIR} build-llama)

# Create our JNI bridge library
add_library(flutter_llama_bridge SHARED
    flutter_llama_bridge.cpp
)

# Include directories
target_include_directories(flutter_llama_bridge PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/src
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/common
)

# Link with llama.cpp libraries
target_link_libraries(flutter_llama_bridge
    llama
    ${log-lib}
    ${android-lib}
)

# Link GPU libraries
if(GGML_OPENCL AND opencl-lib)
    target_link_libraries(flutter_llama_bridge ${opencl-lib})
    message(STATUS "  Linked: OpenCL library (Adreno)")
endif()
if(GGML_VULKAN AND vulkan-lib)
    target_link_libraries(flutter_llama_bridge ${vulkan-lib})
    message(STATUS "  Linked: Vulkan library")
endif()

# ── Multimodal (mtmd) bridge ────────────────────────────────────────
set(MTMD_DIR "${LLAMA_CPP_DIR}/tools/mtmd")

# Collect mtmd core + model sources
set(MTMD_SOURCES
    ${MTMD_DIR}/clip.cpp
    ${MTMD_DIR}/mtmd.cpp
    ${MTMD_DIR}/mtmd-audio.cpp
    ${MTMD_DIR}/mtmd-helper.cpp
    # model-specific clip implementations
    ${MTMD_DIR}/models/cogvlm.cpp
    ${MTMD_DIR}/models/conformer.cpp
    ${MTMD_DIR}/models/glm4v.cpp
    ${MTMD_DIR}/models/internvl.cpp
    ${MTMD_DIR}/models/kimivl.cpp
    ${MTMD_DIR}/models/llama4.cpp
    ${MTMD_DIR}/models/llava.cpp
    ${MTMD_DIR}/models/minicpmv.cpp
    ${MTMD_DIR}/models/mobilenetv5.cpp
    ${MTMD_DIR}/models/pixtral.cpp
    ${MTMD_DIR}/models/qwen2vl.cpp
    ${MTMD_DIR}/models/qwen3vl.cpp
    ${MTMD_DIR}/models/siglip.cpp
    ${MTMD_DIR}/models/whisper-enc.cpp
    ${MTMD_DIR}/models/youtuvl.cpp
)

add_library(flutter_llama_multimodal_bridge SHARED
    flutter_llama_multimodal_bridge.cpp
    ${MTMD_SOURCES}
)

target_include_directories(flutter_llama_multimodal_bridge PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/src
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/common
    ${MTMD_DIR}
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}
)

target_link_libraries(flutter_llama_multimodal_bridge
    llama
    ggml
    ${log-lib}
    ${android-lib}
)

if(GGML_OPENCL AND opencl-lib)
    target_link_libraries(flutter_llama_multimodal_bridge ${opencl-lib})
endif()
if(GGML_VULKAN AND vulkan-lib)
    target_link_libraries(flutter_llama_multimodal_bridge ${vulkan-lib})
endif()

message(STATUS "Flutter Llama Bridge configuration:")
message(STATUS "  Android ABI: ${ANDROID_ABI}")
message(STATUS "  llama.cpp Dir: ${LLAMA_CPP_DIR}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Multimodal: ENABLED (mtmd sources from ${MTMD_DIR})")

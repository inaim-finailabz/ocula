# Flutter Llama - Windows CMake configuration
# Builds llama.cpp from source and links it into the Flutter plugin DLL.

cmake_minimum_required(VERSION 3.14)

set(PLUGIN_NAME "flutter_llama_plugin")
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")

project(${PLUGIN_NAME} LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ─── llama.cpp build options ───────────────────────────────────────────────────

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)  # static libs for simpler linking

set(GGML_METAL   OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP  OFF CACHE BOOL "" FORCE)  # avoid OpenMP runtime dep
set(GGML_AMX     ON  CACHE BOOL "" FORCE)  # Intel AMX (Alder Lake+)
set(GGML_AVX     ON  CACHE BOOL "" FORCE)
set(GGML_AVX2    ON  CACHE BOOL "" FORCE)
set(GGML_F16C    ON  CACHE BOOL "" FORCE)
set(GGML_FMA     ON  CACHE BOOL "" FORCE)

# ─── CUDA (optional) ──────────────────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "CUDA FOUND — GPU acceleration ENABLED")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    message(STATUS "CUDA NOT found — CPU-only build")
endif()

# ─── Vulkan (optional) ────────────────────────────────────────────────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "Vulkan FOUND — GPU acceleration ENABLED")
else()
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()

# ─── Build llama.cpp ──────────────────────────────────────────────────────────
add_subdirectory(${LLAMA_CPP_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build-llama")

# ─── Plugin shared library ────────────────────────────────────────────────────
add_library(${PLUGIN_NAME} SHARED
    "flutter_llama_plugin.cpp"
    "flutter_llama_plugin_c_api.cpp"
    "llama_cpp_bridge_win.cpp"
)

# Flutter Windows plugin boilerplate — registers the plugin entry point
set_target_properties(${PLUGIN_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    DEFINE_SYMBOL FLUTTER_PLUGIN_IMPL
)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL _UNICODE UNICODE)

# Flutter SDK headers (injected by the build system via FLUTTER_ROOT)
target_include_directories(${PLUGIN_NAME} PRIVATE
    "${FLUTTER_MANAGED_DIR}/ephemeral/include"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/src"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${PLUGIN_NAME} PRIVATE
    flutter
    flutter_wrapper_plugin
    llama
    ggml
)

if(GGML_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(${PLUGIN_NAME} PRIVATE cuda cudart)
endif()
if(GGML_VULKAN AND Vulkan_FOUND)
    target_link_libraries(${PLUGIN_NAME} PRIVATE Vulkan::Vulkan)
endif()

# Bundled libraries that must ship next to the .exe
set(flutter_llama_plugin_bundled_libraries
    $<TARGET_FILE:${PLUGIN_NAME}>
    PARENT_SCOPE
)

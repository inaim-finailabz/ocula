cmake_minimum_required(VERSION 3.16)
project(ocula_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- llama.cpp dependency ---
# Clone llama.cpp into native/vendor/llama.cpp before building:
#   git clone https://github.com/ggerganov/llama.cpp vendor/llama.cpp
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp)

if(EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    add_subdirectory(${LLAMA_DIR} llama_build)
    set(LLAMA_AVAILABLE ON)
else()
    message(WARNING "llama.cpp not found at ${LLAMA_DIR}. Building bridge stub only.")
    set(LLAMA_AVAILABLE OFF)
endif()

# --- Bridge shared library ---
add_library(ocula_bridge SHARED bridge.cpp)

target_compile_options(ocula_bridge PRIVATE -Wall -Wextra -O2)

if(LLAMA_AVAILABLE)
    target_link_libraries(ocula_bridge PRIVATE llama common)
    target_compile_definitions(ocula_bridge PRIVATE LLAMA_AVAILABLE=1)
endif()

# --- Platform-specific settings ---
if(ANDROID)
    # Android NDK builds
    target_compile_definitions(ocula_bridge PRIVATE PLATFORM_ANDROID=1)
elseif(APPLE)
    # iOS / macOS — enable Metal acceleration
    target_compile_definitions(ocula_bridge PRIVATE PLATFORM_APPLE=1)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK)
        target_link_libraries(ocula_bridge PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()

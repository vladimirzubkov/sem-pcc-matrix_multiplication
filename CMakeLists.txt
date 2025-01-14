cmake_minimum_required(VERSION 3.29)
project(sem)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the common output directory for all build types
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Separate output directories for Debug and Release
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/Release)

# Disable in-source builds (force out-of-source builds)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

# Suppress automatic regeneration of Makefiles
set(CMAKE_SUPPRESS_REGENERATION ON)

# Generate compile_commands.json for better editor integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the executable target
add_executable(sem main.cpp)

# Set output names for Debug and Release builds
set_target_properties(sem PROPERTIES
        RUNTIME_OUTPUT_NAME_DEBUG "sem-debug"
        RUNTIME_OUTPUT_NAME_RELEASE "sem-release"
)

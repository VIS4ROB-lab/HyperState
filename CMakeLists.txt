# Check CMake version.
cmake_minimum_required(VERSION 3.16)
enable_language(CXX)

# Includes.
include(FetchContent)

# Define projects.
project(HyperState VERSION 22.07.01 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperState LANGUAGES CXX)

# Define libraries and aliases.
add_library(${PROJECT_NAME} SHARED)
add_library(Hyper::State ALIAS ${PROJECT_NAME})

# Set properties.
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_compile_options(${PROJECT_NAME} PUBLIC -O3 -march=native -fconcepts)

# Fetch (external) content.
FetchContent_Declare(HyperVariables GIT_REPOSITORY https://github.com/VIS4ROB-lab/HyperVariables.git GIT_TAG 81b9606a161c259422a24c63c3e57868d26f14ed)
FetchContent_MakeAvailable(HyperVariables)

# Find Eigen target if not present.
if (NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED NO_MODULE)
endif ()

# Find GTest target if not present.
if (NOT TARGET GTest::GTest)
    find_package(GTest REQUIRED)
endif ()

# Find glog target if not present.
if (NOT TARGET glog::glog)
    find_package(glog REQUIRED)
endif ()

# Add includes and link libraries.
file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/internal/*.hpp ${PROJECT_SOURCE_DIR}/internal/*.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Hyper::Variables)

# Build tests.
option(HYPER_BUILD_TESTING "Build tests." TRUE)
if (HYPER_BUILD_TESTING)
    add_subdirectory(${PROJECT_SOURCE_DIR}/apps/tests)
endif (HYPER_BUILD_TESTING)

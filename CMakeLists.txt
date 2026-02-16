cmake_minimum_required(VERSION 3.22)
project(VideoProcessor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Video libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
find_package(OnnxRuntime QUIET)
if (NOT OnnxRuntime_FOUND)
    find_library(ORT_LIB onnxruntime)
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime core/session)

    if (ORT_LIB AND ORT_INCLUDE_DIR)
        add_library(OnnxRuntime UNKNOWN IMPORTED)
        set_target_properties(OnnxRuntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        )
        message(STATUS "Found OnnxRuntime (Manual): ${ORT_LIB}")
    else()
        message(STATUS "OnnxRuntime not found. Fetching prebuilt binary...")
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
        )
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        
        set(ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
        # Identify lib path - might be lib/ or lib64/
        if(EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
            set(ORT_LIB "${ORT_ROOT}/lib/libonnxruntime.so")
        else()
             file(GLOB ORT_LIBS "${ORT_ROOT}/lib/libonnxruntime.so*")
             list(GET ORT_LIBS 0 ORT_LIB)
        endif()
        set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
        
        add_library(OnnxRuntime UNKNOWN IMPORTED)
        set_target_properties(OnnxRuntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        )
        include_directories(${ORT_INCLUDE_DIR})
    endif()
else()
    add_library(OnnxRuntime ALIAS OnnxRuntime::OnnxRuntime)
endif()

# Definitions for YOLO backend
add_compile_definitions(_YOLO_ONNXRUNTIME _YOLO_OPENCV)

# Eigen
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)

# Include directories
include_directories(
    src/yolo
    src/yolo/onnxruntime
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Source files
file(GLOB YOLO_SRCS "src/yolo/*.cpp" "src/yolo/onnxruntime/*.cpp")
set(SRCS
    src/main.cpp
    src/VideoProcessor.cpp
    ${YOLO_SRCS}
)

add_executable(video_processor ${SRCS})

target_link_libraries(video_processor
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OnnxRuntime
    stdc++fs
)

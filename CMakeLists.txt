cmake_minimum_required(VERSION 3.0.0)
project(vscodetest VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(lib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(segmentation "lib/HOG.cpp" "lib/Utility.cpp" "lib/SvmOps.cpp" "lib/ImgOps.cpp" segmentation.cpp "lib/RectClustering.cpp" "lib/PreProcessing.cpp" "lib/Metrics.cpp" "lib/detectionKmeans.cpp")
add_executable(detection "lib/HOG.cpp" "lib/Utility.cpp" "lib/SvmOps.cpp" "lib/ImgOps.cpp" detection.cpp "lib/RectClustering.cpp" "lib/PreProcessing.cpp" "lib/Metrics.cpp" "lib/detectionKmeans.cpp")

target_link_libraries( segmentation ${OpenCV_LIBS})
target_link_libraries( detection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

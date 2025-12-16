cmake_minimum_required(VERSION 3.10)

# Project name
project(svm-learning)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# ---- Training executable ----
add_executable(train_svm
    src/train_svm.cpp
)

target_link_libraries(train_svm
    ${OpenCV_LIBS}
)

# ---- Monitoring executable ----
add_executable(monitor
    src/monitor.cpp
)

target_link_libraries(monitor
    ${OpenCV_LIBS}
)

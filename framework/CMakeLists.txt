#
#
# Practical Course: GPU Programming in Computer Vision
#
#
# Technical University Munich, Computer Vision Group
# Winter Semester 2015/2016, March 15 - April 15
#
#

PROJECT(Exercises)
SET(CUDA_SEPARABLE_COMPILATION ON)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)


# CUDA Flags
SET(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	--debug; # Create debug information
	--device-debug; # Create debug information for device
	#--profile;
	#-lineinfo; # Add line information 
	-Wno-deprecated-gpu-targets; # Disable the warning
	-rdc=true; # Allow relocation of device code -> CUDA_SEPARABLE_COMPILATION On
)

# message("CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}")

# Files for different exercises
SET(COMMON_SRC src/util.cu src/usage.cu)
SET(IMAGE_SRC src/image.cu)

# CUDA include directories
INCLUDE_DIRECTORIES(
	include/
)


# Main target
CUDA_ADD_EXECUTABLE(
	depth_adaptive_diffusion
	main.cu
	${COMMON_SRC}
	${IMAGE_SRC}
)

TARGET_LINK_LIBRARIES(depth_adaptive_diffusion ${OpenCV_LIBS})
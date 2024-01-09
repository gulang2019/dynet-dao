cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(dao_actual_memory LANGUAGES CUDA CXX)
enable_language(CUDA)

function(find_cudnn)
  if (DEFINED ENV{CUDNN_ROOT} AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT $ENV{CUDNN_ROOT})
  elseif (DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
#  set(CUDNN_ROOT /usr/local/cuda CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    HINTS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so cudnn.lib
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
          ${CUDNN_ROOT}/lib/x64
    DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(STATUS "Failed to find CUDNN in path: ${CUDNN_ROOT} (Did you set CUDNN_ROOT properly?)")
  endif()
endfunction()

find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
#list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
list(APPEND CUDA_LIBRARIES ${CUDA_curand_LIBRARY})
MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
set(LIBS ${LIBS} ${CUDA_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_GPU")
find_cudnn()
if(CUDNN_FOUND)
  include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
  message("-- Successfully include CUDNN flags")
else()
  message("-- CUDNN not found, some dependent functionalities will be disabled")
endif()
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)
message("-- CUDA_NVTX_LIBRARY: ${CUDA_NVTX_LIBRARY}")
if(CUDA_NVTX_LIBRARY)
  list(APPEND CUDA_LIBRARIES ${CUDA_NVTX_LIBRARY})
  set(LIBS ${LIBS} ${CUDA_NVTX_LIBRARY})  
endif()



file(GLOB_RECURSE SOURCES *.cc)
file(GLOB_RECURSE HEADERS *.h)


add_executable(dao_actual_memory ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
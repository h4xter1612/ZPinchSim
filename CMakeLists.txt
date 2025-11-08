cmake_minimum_required(VERSION 3.16)
project(ZPinchStabilitySim LANGUAGES CXX)

# ---------- Global configuration ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

if (MSVC)
    add_compile_options(/W4 /permissive- /bigobj)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------- Sources ----------
file(GLOB SRC_ALL "${CMAKE_SOURCE_DIR}/src/*.cpp")
# Exclude main.cpp from the static library
list(FILTER SRC_ALL EXCLUDE REGEX ".*/main\\.cpp$")

# Exclude legacy files that conflict with toy MHD
list(FILTER SRC_ALL EXCLUDE REGEX ".*/riemann\\.cpp$")
list(FILTER SRC_ALL EXCLUDE REGEX ".*/time_integrator\\.cpp$")

set(MAIN_CPP "${CMAKE_SOURCE_DIR}/src/main.cpp")
set(TEST_GRID_CPP "${CMAKE_SOURCE_DIR}/src/test_grid.cpp")

# ---------- Base library ----------
add_library(zpinch STATIC ${SRC_ALL})
target_include_directories(zpinch PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---------- Executable ----------
add_executable(zpinch_run ${MAIN_CPP})
target_link_libraries(zpinch_run PRIVATE zpinch)

# ---------- OpenMP ----------
# On MSVC prefer the LLVM OpenMP frontend to get modern features.
if (MSVC)
  # Do NOT link OpenMP::OpenMP_CXX (it injects /openmp clásico).
  target_compile_options(zpinch PUBLIC /openmp:llvm)
  target_link_options(zpinch  PUBLIC /openmp:llvm)
  target_compile_options(zpinch_run PRIVATE /openmp:llvm)
  target_link_options(zpinch_run  PRIVATE /openmp:llvm)

  if (EXISTS "${TEST_GRID_CPP}")
    target_compile_options(test_grid PRIVATE /openmp:llvm)
    target_link_options(test_grid  PRIVATE /openmp:llvm)
  endif()
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(zpinch PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(zpinch_run PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------- Optional test executable ----------
if (EXISTS "${TEST_GRID_CPP}")
    add_executable(test_grid ${TEST_GRID_CPP})
    target_link_libraries(test_grid PRIVATE zpinch)
    set_target_properties(test_grid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()

# ---------- MSVC clean runtime output ----------
set_target_properties(zpinch_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# ---------- Copy assets (configs and pyscripts) next to the exe ----------
file(GLOB CONFIG_FILES
    "${CMAKE_SOURCE_DIR}/configs/*.yaml"
    "${CMAKE_SOURCE_DIR}/configs/*.yml"
)
file(GLOB PYSCRIPTS_FILES
    "${CMAKE_SOURCE_DIR}/pyscripts/*.py"
)

set(RUN_DIR "$<TARGET_FILE_DIR:zpinch_run>")
set(CONF_DST "${RUN_DIR}/configs")
set(PYSCRIPTS_DST  "${RUN_DIR}/pyscripts")
set(DATA_DST "${RUN_DIR}/data")
set(DEBUG_DST "${DATA_DST}/debug")

add_custom_command(TARGET zpinch_run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CONF_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PYSCRIPTS_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DATA_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DEBUG_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CONFIG_FILES} "${CONF_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYSCRIPTS_FILES} "${PYSCRIPTS_DST}"
    VERBATIM
)

# ---------- Default Release for single-config generators ----------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")


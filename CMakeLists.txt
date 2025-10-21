cmake_minimum_required(VERSION 3.16)
project(ZPinchStabilitySim LANGUAGES CXX)

# ---------- Configuración global ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

if (MSVC)
    add_compile_options(/W4 /permissive- /bigobj)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------- Fuentes ----------
file(GLOB SRC_ALL "${CMAKE_SOURCE_DIR}/src/*.cpp")
# Excluir el main.cpp del listado para la librería
list(FILTER SRC_ALL EXCLUDE REGEX ".*/main\\.cpp$")

# === EXCLUIR LEGACY QUE CHOCAN CON EL TOY MHD (Paso 4) ===
# riemann.cpp y time_integrator.cpp antiguos usan API/namespace distintos
list(FILTER SRC_ALL EXCLUDE REGEX ".*/riemann\\.cpp$")
list(FILTER SRC_ALL EXCLUDE REGEX ".*/time_integrator\\.cpp$")

set(MAIN_CPP "${CMAKE_SOURCE_DIR}/src/main.cpp")
set(TEST_GRID_CPP "${CMAKE_SOURCE_DIR}/src/test_grid.cpp")

# ---------- Librería base ----------
add_library(zpinch STATIC ${SRC_ALL})
target_include_directories(zpinch PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---------- Ejecutable principal ----------
add_executable(zpinch_run ${MAIN_CPP})
target_link_libraries(zpinch_run PRIVATE zpinch)

# ---------- Ejecutable de pruebas (opcional) ----------
if (EXISTS "${TEST_GRID_CPP}")
    add_executable(test_grid ${TEST_GRID_CPP})
    target_link_libraries(test_grid PRIVATE zpinch)
    # Salida “limpia” también para test_grid
    set_target_properties(test_grid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()

# ---------- Salidas “limpias” en MSVC ----------
set_target_properties(zpinch_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# ---------- Copia de assets (configs y viz) al lado del ejecutable ----------
file(GLOB CONFIG_FILES
    "${CMAKE_SOURCE_DIR}/configs/*.yaml"
    "${CMAKE_SOURCE_DIR}/configs/*.yml"
)
file(GLOB VIZ_FILES
    "${CMAKE_SOURCE_DIR}/viz/*.py"
)

set(RUN_DIR "$<TARGET_FILE_DIR:zpinch_run>")
set(CONF_DST "${RUN_DIR}/configs")
set(VIZ_DST  "${RUN_DIR}/viz")
set(DATA_DST "${RUN_DIR}/data")
set(DEBUG_DST "${DATA_DST}/debug")

add_custom_command(TARGET zpinch_run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CONF_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VIZ_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DATA_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DEBUG_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CONFIG_FILES} "${CONF_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VIZ_FILES} "${VIZ_DST}"
    VERBATIM
)

# ---------- Modo Release por defecto (solo single-config) ----------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")


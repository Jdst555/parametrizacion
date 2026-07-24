cmake_minimum_required(VERSION 3.16)
project(Harmonic)

# --- NEW: Default to Release mode if not specified ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# 1. Force Static Library Mode
set(LIBIGL_USE_STATIC_LIBRARY ON CACHE BOOL "Use libigl as static library" FORCE)
set(LIBIGL_SOURCE_DIR "C:/Users/jasaa/Documents/TFM_WORKBENCH/libigl-main")

# 2. Add the libigl directory
add_subdirectory(${LIBIGL_SOURCE_DIR} libigl)

# 3. TELL LIBIGL TO BUILD THE VIEWER (The new v2.4+ way)
igl_include(glfw)
igl_include(imgui)

# 4. Add your executable
add_executable(${PROJECT_NAME} app.cpp)


# 5. C++ Standard
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- NEW: Copy camelhead.off to the executable directory ---
# This ensures that no matter where CMake puts the executable (e.g., in a /Release subfolder), 
# the .off file will be right next to it.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/camelhead.off"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/camelhead.off"
    COMMENT "Copying camelhead.off to the executable directory"
)

# 7. Link against igl::core and the newly renamed igl::glfw target
target_link_libraries(${PROJECT_NAME} PRIVATE igl::core igl::glfw igl::imgui)
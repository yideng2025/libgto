cmake_minimum_required(VERSION 3.16)
project(libgto VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(LIBGTO_ENABLE_OPENMP "Enable task parallelism" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Boost 1.75 REQUIRED)

add_library(gto
  src/api.cpp src/prepared.cpp src/plan.cpp src/workspace.cpp
  src/executor.cpp src/output.cpp src/md.cpp src/reference.cpp
  src/sensitive.cpp src/angular_products.cpp src/high_angular.cpp)
add_library(libgto::gto ALIAS gto)
target_include_directories(gto SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(gto PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(LIBGTO_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(gto PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(gto PRIVATE -fno-fast-math -ffp-contract=off)
endif()
set_target_properties(gto PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

install(TARGETS gto EXPORT libgtoTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/libgto.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT libgtoTargets NAMESPACE libgto::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libgto)
configure_package_config_file(cmake/libgtoConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libgtoConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libgto)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/libgtoConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgtoConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/libgtoConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libgto)

# Ordinary builds use checked-in constants; this target reproduces their source.
add_executable(libgto_generate_boys EXCLUDE_FROM_ALL tools/generate_boys.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(libgto_generate_boys PRIVATE -fno-fast-math -ffp-contract=off)
endif()
add_custom_target(libgto_regenerate_boys
  COMMAND $<TARGET_FILE:libgto_generate_boys> ${CMAKE_CURRENT_SOURCE_DIR}/src/boys_table.hpp
  DEPENDS libgto_generate_boys VERBATIM)

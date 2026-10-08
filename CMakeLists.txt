cmake_minimum_required(VERSION 3.16)
project(maestro C CXX)

# ==============================================================================
# 1. Global Project Settings & Defaults
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Global binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

# Fast tests definition across all targets
add_definitions(-D_FAST_TESTS)

# Optional ccache launcher for compilation speedup
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    message(STATUS "ccache found, using it for compilation")
else()
    message(STATUS "ccache not found, install it for faster recompilation: brew install ccache")
endif()

# Suppress deprecation warning on CMake >= 3.30 for FetchContent_Populate
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
# Keep FindBoost module active so BOOST_ROOT works for custom/local Boost installations
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
# Suppress CMP0146 (FindCUDA) deprecation warning in third-party dependencies (e.g. Eigen)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress deprecation warnings" FORCE)
# Honor uppercase <PACKAGENAME>_ROOT variables (e.g., BOOST_ROOT)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

include(FetchContent)

# ==============================================================================
# 2. Build Options
# ==============================================================================

option(COMPILE_TESTS "Compile test suite" ON)
if(NOT "$ENV{NO_TESTS}" STREQUAL "")
    set(COMPILE_TESTS OFF)
endif()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(MAESTRO_BUILD_MPI_GPU_TESTS "Build MPI GPU hardware tests (only tests link MPI)" OFF)
option(MAESTRO_ENABLE_OPENMP "Enable OpenMP acceleration" ON)
option(MAESTRO_ENABLE_COMPOSER "Enable integration with Composer" OFF)

if(MAESTRO_ENABLE_COMPOSER)
    find_path(COMPOSER_INCLUDE_DIR
        NAMES composer/Estimators/ExecutionEstimator.h
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../composer" "${CMAKE_CURRENT_SOURCE_DIR}/../../composer"
        DOC "Composer source root (the directory containing composer/)")
    if(NOT EXISTS "${COMPOSER_INCLUDE_DIR}/composer/Estimators/ExecutionEstimator.h" OR
       NOT EXISTS "${COMPOSER_INCLUDE_DIR}/composer/Network/SimpleFullyConnectedNetwork.h")
        message(FATAL_ERROR
            "MAESTRO_ENABLE_COMPOSER requires Composer headers. "
            "Set COMPOSER_INCLUDE_DIR to the Composer source root (containing composer/).")
    endif()
    add_library(maestro_composer INTERFACE)
    target_compile_definitions(maestro_composer INTERFACE MAESTRO_ENABLE_COMPOSER)
    # Composer headers also include Maestro headers relative to its source root.
    target_include_directories(maestro_composer INTERFACE
        "${COMPOSER_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Qiskit Aer simulator backend (enabled by default unless explicitly disabled)
option(ENABLE_QISKIT_AER "Enable Qiskit Aer simulator backend" ON)
if(NOT "$ENV{NO_QISKIT_AER}" STREQUAL "" OR NOT ENABLE_QISKIT_AER)
    add_definitions(-DNO_QISKIT_AER)
    set(QAER False)
    message(STATUS "Qiskit Aer support explicitly disabled")
else()
    set(QAER True)
    message(STATUS "Qiskit Aer support enabled")
endif()

# ==============================================================================
# 3. Compiler Flags & Architecture / SIMD
# ==============================================================================

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /bigobj")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds -Wno-delete-non-abstract-non-virtual-dtor")
endif()

include(CheckCXXCompilerFlag)
function(enable_cxx_flag flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_set)
    if(flag_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

# Architecture SIMD flags
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR
   CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64" OR
   CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
    if(UNIX OR APPLE)
        if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            message(STATUS "Setting SIMD flags for x86_64...")
            set(SIMD_FLAGS "-mfma;-mavx2")
            enable_cxx_flag("-mpopcnt")
        endif()
    elseif(MSVC)
        message(STATUS "Setting SIMD flags for MSVC...")
        set(SIMD_FLAGS "/arch:AVX2")
    endif()
endif()

# ==============================================================================
# 4. External Dependencies
# ==============================================================================

# --- Eigen 5 (header-only, auto-fetched if not provided) ---
if(NOT DEFINED EIGEN5_INCLUDE_DIR)
    set(EIGEN5_INCLUDE_DIR "$ENV{EIGEN5_INCLUDE_DIR}")
    if(NOT EIGEN5_INCLUDE_DIR)
        message(STATUS "EIGEN5_INCLUDE_DIR not set, fetching Eigen 5.0.0...")
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(eigen)
        set(EIGEN5_INCLUDE_DIR ${eigen_SOURCE_DIR})
        message(STATUS "Eigen fetched to: ${EIGEN5_INCLUDE_DIR}")
    endif()
endif()

# --- QCSim (source fetched, requires FFTW3 if compiled) ---
if(NOT DEFINED QCSIM_INCLUDE_DIR)
    set(QCSIM_INCLUDE_DIR "$ENV{QCSIM_INCLUDE_DIR}")
    if(NOT QCSIM_INCLUDE_DIR)
        message(STATUS "QCSIM_INCLUDE_DIR not set, fetching QCSim...")
        FetchContent_Declare(
            qcsim
            GIT_REPOSITORY https://github.com/QoroQuantum/QCSim.git
            GIT_TAG 217c60953795045ecb6439d0cc34191fecd165ff
        )
        FetchContent_GetProperties(qcsim)
        if(NOT qcsim_POPULATED)
            FetchContent_Populate(qcsim)
        endif()
        set(QCSIM_INCLUDE_DIR ${qcsim_SOURCE_DIR}/QCSim)
        message(STATUS "QCSim fetched to: ${QCSIM_INCLUDE_DIR}")
    endif()
endif()

# --- Optional Qiskit Aer Dependencies ---
if(QAER)
    if(NOT DEFINED JSON_INCLUDE_DIR)
        set(JSON_INCLUDE_DIR "$ENV{JSON_INCLUDE_DIR}")
        if(NOT JSON_INCLUDE_DIR)
            message(STATUS "JSON_INCLUDE_DIR not set, fetching nlohmann/json...")
            FetchContent_Declare(
                json
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG v3.11.2
            )
            FetchContent_GetProperties(json)
            if(NOT json_POPULATED)
                FetchContent_Populate(json)
            endif()
            set(JSON_INCLUDE_DIR ${json_SOURCE_DIR}/single_include)
            message(STATUS "nlohmann/json fetched to: ${JSON_INCLUDE_DIR}")
        endif()
    endif()

    if(NOT DEFINED AER_INCLUDE_DIR OR AER_INCLUDE_DIR STREQUAL "")
        set(AER_INCLUDE_DIR "$ENV{AER_INCLUDE_DIR}")
    endif()
    if(AER_INCLUDE_DIR)
        file(TO_CMAKE_PATH "${AER_INCLUDE_DIR}" AER_INCLUDE_DIR)
        if(EXISTS "${AER_INCLUDE_DIR}/src/framework/linalg/vector.hpp")
            set(AER_INCLUDE_DIR "${AER_INCLUDE_DIR}/src")
        endif()
    endif()

    if(NOT AER_INCLUDE_DIR OR NOT EXISTS "${AER_INCLUDE_DIR}/framework/linalg/vector.hpp")
        message(STATUS "AER_INCLUDE_DIR not set or does not contain framework/linalg/vector.hpp (${AER_INCLUDE_DIR}), fetching Qiskit Aer from QoroQuantum/qiskit-aer...")
        FetchContent_Declare(
            qiskit_aer
            GIT_REPOSITORY https://github.com/QoroQuantum/qiskit-aer.git
            GIT_TAG 80bd205b8b224e6db7d8c651473fda5cf346426c
        )
        FetchContent_GetProperties(qiskit_aer)
        if(NOT qiskit_aer_POPULATED)
            FetchContent_Populate(qiskit_aer)
        endif()
        set(AER_INCLUDE_DIR ${qiskit_aer_SOURCE_DIR}/src)
        message(STATUS "Qiskit Aer fetched to: ${AER_INCLUDE_DIR}")
    else()
        message(STATUS "Using Qiskit Aer include directory: ${AER_INCLUDE_DIR}")
    endif()

    if(APPLE AND NOT OpenBLAS_ROOT AND "$ENV{OpenBLAS_ROOT}" STREQUAL "")
        find_program(BREW_CMD brew)
        if(BREW_CMD)
            execute_process(COMMAND ${BREW_CMD} --prefix openblas
                            OUTPUT_VARIABLE OPENBLAS_BREW_PREFIX
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(OPENBLAS_BREW_PREFIX)
                list(PREPEND CMAKE_PREFIX_PATH "${OPENBLAS_BREW_PREFIX}")
                set(OpenBLAS_ROOT "${OPENBLAS_BREW_PREFIX}")
            endif()
        endif()
    elseif(WIN32)
        if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-windows")
            list(PREPEND CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows")
        elseif(EXISTS "C:/vcpkg/installed/x64-windows")
            list(PREPEND CMAKE_PREFIX_PATH "C:/vcpkg/installed/x64-windows")
        endif()
    endif()

    if(NOT DEFINED BLA_VENDOR AND "$ENV{BLA_VENDOR}" STREQUAL "" AND NOT BLAS_LIBRARIES)
        # 1. Try finding OpenBLAS via modern CMake CONFIG mode (standard for vcpkg on Windows and modern distributions)
        find_package(OpenBLAS CONFIG QUIET)
        if(OpenBLAS_FOUND)
            message(STATUS "Found OpenBLAS via CONFIG mode")
            set(BLAS_FOUND TRUE)
            if(TARGET OpenBLAS::OpenBLAS)
                set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
            elseif(OpenBLAS_LIBRARIES)
                set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
            else()
                set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
            endif()
            if(OpenBLAS_INCLUDE_DIRS)
                include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIRS})
            elseif(OpenBLAS_INCLUDE_DIR)
                include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
            endif()
            message(STATUS "Using OpenBLAS: ${BLAS_LIBRARIES}")
        else()
            # 2. Try FindBLAS with OpenBLAS vendor
            set(BLA_VENDOR OpenBLAS)
            find_package(BLAS QUIET)
            if(NOT BLAS_FOUND)
                message(STATUS "OpenBLAS not found, falling back to the generic BLAS search")
                unset(BLA_VENDOR)
                unset(BLA_VENDOR CACHE)
                unset(BLAS_LIBRARIES)
                unset(BLAS_LIBRARIES CACHE)
                unset(BLAS_FOUND)
            else()
                message(STATUS "Using OpenBLAS: ${BLAS_LIBRARIES}")
            endif()
        endif()
    endif()

    if(NOT BLAS_FOUND)
        find_package(BLAS REQUIRED)
        if(NOT BLAS_FOUND)
            message(FATAL_ERROR "Couldn't find BLAS libraries")
        endif()
    endif()

    set(MAESTRO_LINALG_LIBRARIES ${BLAS_LIBRARIES})
    if(WIN32)
        # vcpkg builds OpenBLAS without LAPACK. Aer also needs the LAPACK
        # eigensolver and SVD routines, provided by lapack-reference.
        find_package(lapack CONFIG REQUIRED)
        if(NOT TARGET lapack)
            message(FATAL_ERROR "The LAPACK package must provide the lapack target")
        endif()
        list(PREPEND MAESTRO_LINALG_LIBRARIES lapack)
    endif()
endif()

# --- Boost ---
if(NOT "$ENV{BOOST_ROOT}" STREQUAL "")
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    set(Boost_ROOT "$ENV{BOOST_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
    message(STATUS "Using Boost from BOOST_ROOT: ${BOOST_ROOT}")
endif()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_DEBUG_LIBS OFF)
find_package(Boost REQUIRED COMPONENTS program_options json container serialization)
if(MSVC AND NOT Boost_USE_STATIC_LIBS)
    # Match MSVC's Boost auto-linking to the shared libraries selected above.
    link_libraries(Boost::dynamic_linking Boost::disable_autolinking)
endif()

if(NOT Boost_FOUND)
    message(FATAL_ERROR "\n"
        "Boost library not found!\n"
        "Please install Boost 1.89+ with the following components:\n"
        "  - program_options\n"
        "  - json\n"
        "  - container\n"
        "  - serialization\n"
        "\n"
        "Options:\n"
        "  1. Use build.sh script (builds Boost locally):\n"
        "     ./build.sh\n"
        "\n"
        "  2. Install Boost system-wide:\n"
        "     Ubuntu/Debian: sudo apt-get install libboost-all-dev\n"
        "     Fedora/RHEL:   sudo dnf install boost-devel\n"
        "     macOS:         brew install boost\n"
        "     Conda:         conda install -c conda-forge boost\n"
        "\n"
        "  3. Set BOOST_ROOT environment variable to point to your Boost installation\n"
        "\n"
        "See INSTALL.md for more details.\n"
    )
endif()

# System Boost may report /usr/include. Adding compiler-default directories
# with -isystem can break libstdc++'s #include_next lookup of C headers.
set(MAESTRO_DEPENDENCY_INCLUDE_DIRS
    ${EIGEN5_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QCSIM_INCLUDE_DIR})
if(QAER)
    list(APPEND MAESTRO_DEPENDENCY_INCLUDE_DIRS ${JSON_INCLUDE_DIR} ${AER_INCLUDE_DIR})
endif()
list(REMOVE_ITEM MAESTRO_DEPENDENCY_INCLUDE_DIRS ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${MAESTRO_DEPENDENCY_INCLUDE_DIRS})

foreach(boost_target Boost::headers Boost::boost Boost::container Boost::json
                     Boost::serialization Boost::program_options)
    if(TARGET ${boost_target})
        get_target_property(boost_include_target ${boost_target} ALIASED_TARGET)
        if(NOT boost_include_target)
            set(boost_include_target ${boost_target})
        endif()
        foreach(include_property INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            get_target_property(boost_target_includes ${boost_include_target} ${include_property})
            if(boost_target_includes)
                list(REMOVE_ITEM boost_target_includes ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
                set_property(TARGET ${boost_include_target} PROPERTY ${include_property} "${boost_target_includes}")
            endif()
        endforeach()
    endif()
endforeach()

# --- OpenMP ---
if(MSVC)
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "/openmp")
    set(OpenMP_CXX_LIB_NAMES "")
endif()

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
endif()

if(MAESTRO_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
else()
    message(STATUS "OpenMP support disabled via MAESTRO_ENABLE_OPENMP")
    set(OpenMP_CXX_FOUND FALSE CACHE BOOL "OpenMP disabled" FORCE)
endif()

# ==============================================================================
# 5. Core Targets: libmaestro and maestroexe
# ==============================================================================

set(MAESTROSRC maestrolib/Interface.cpp Execution/Request.cpp)
if(WIN32)
    list(APPEND MAESTROSRC maestrolib/DllMain.cpp)
else()
    list(APPEND MAESTROSRC Simulators/Core/Factory.cpp)
endif()

set(MAESTROEXESRC maestroexe/MaestroExe.cpp)

# Define core library and executable
add_library(maestro SHARED ${MAESTROSRC})
if(MAESTRO_ENABLE_COMPOSER)
    target_link_libraries(maestro PUBLIC maestro_composer)
endif()
# Build identity is informational; schema_version remains the protocol contract.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" MAESTRO_VERSION_LINE REGEX "^version = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" MAESTRO_BUILD_VERSION "${MAESTRO_VERSION_LINE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
# Avoid attributing an unpacked source archive to an unrelated parent repository.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND git describe --always --dirty --exclude "*"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE MAESTRO_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT MAESTRO_SOURCE_REVISION)
    set(MAESTRO_SOURCE_REVISION "unknown")
endif()
configure_file(cmake/MaestroBuildInfo.h.in generated/MaestroBuildInfo.h @ONLY)
target_include_directories(maestro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(QAER)
    target_include_directories(maestro PUBLIC ${AER_INCLUDE_DIR} ${JSON_INCLUDE_DIR})
endif()
set_target_properties(maestro PROPERTIES OUTPUT_NAME "maestro")

add_executable(maestroexe ${MAESTROEXESRC})
set_target_properties(maestroexe PROPERTIES OUTPUT_NAME "maestro")

# RPATH configuration
if(APPLE)
    set_target_properties(maestro PROPERTIES INSTALL_NAME_DIR "@rpath")
elseif(UNIX)
    set_target_properties(maestro PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        # Build-tree C consumers also need paths to in-project Boost libraries.
        # CMake applies the installed runtime paths during installation.
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()

# Link dependencies & compile options for maestro and maestroexe
target_link_libraries(maestro PRIVATE Boost::json Boost::serialization)
target_link_libraries(maestroexe PRIVATE Boost::serialization Boost::program_options)

if(UNIX)
    target_link_libraries(maestro PRIVATE dl)
    target_link_libraries(maestroexe PRIVATE dl)
endif()

if(SIMD_FLAGS)
    target_compile_options(maestro PUBLIC ${SIMD_FLAGS})
endif()

if(QAER AND BLAS_FOUND)
    target_link_libraries(maestro PUBLIC ${MAESTRO_LINALG_LIBRARIES})
endif()

# OpenMP target configuration
if(OpenMP_CXX_FOUND)
    message(STATUS "Configuring OpenMP for maestro and maestroexe")
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_program(BREW_CMD brew)
        if(BREW_CMD)
            execute_process(COMMAND ${BREW_CMD} --prefix libomp
                            OUTPUT_VARIABLE LIBOMP_PREFIX
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            message(STATUS "Found libomp at: ${LIBOMP_PREFIX}")
        endif()

        target_compile_options(maestro PUBLIC -Xpreprocessor -fopenmp)

        if(LIBOMP_PREFIX)
            target_include_directories(maestro PUBLIC ${OpenMP_CXX_INCLUDE_DIRS} "${LIBOMP_PREFIX}/include")
            target_link_directories(maestro PUBLIC "${LIBOMP_PREFIX}/lib")
        else()
            target_include_directories(maestro PUBLIC ${OpenMP_CXX_INCLUDE_DIRS} "$ENV{OpenMP_ROOT}/include")
            target_link_directories(maestro PUBLIC "$ENV{OpenMP_ROOT}/lib")
        endif()

        target_link_libraries(maestro PUBLIC ${OpenMP_CXX_LIBRARIES})
    else()
        target_link_libraries(maestro PUBLIC OpenMP::OpenMP_CXX)
        if(NOT MSVC)
            target_link_libraries(maestro PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
        target_compile_options(maestro PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endif()

# ==============================================================================
# 6. Python Bindings (nanobind)
# ==============================================================================

if(BUILD_PYTHON_BINDINGS)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python ${Python_VERSION} (${Python_EXECUTABLE})")

        FetchContent_Declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG        v2.0.0
        )
        FetchContent_MakeAvailable(nanobind)

        nanobind_add_module(maestro_py python/Bindings.cpp)
        set_target_properties(maestro_py PROPERTIES OUTPUT_NAME "maestro")

        if(WIN32)
            set_target_properties(maestro_py PROPERTIES ARCHIVE_OUTPUT_NAME "maestro_py")
        endif()

        target_link_libraries(maestro_py PRIVATE maestro)
        target_include_directories(maestro_py PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/maestrolib
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

        if(APPLE)
            set_target_properties(maestro_py PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../lib"
            )
        elseif(UNIX)
            set_target_properties(maestro_py PROPERTIES
                INSTALL_RPATH "$ORIGIN:$ORIGIN/..:$ORIGIN/../lib"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()

        # Python install rules
        if(WIN32)
            install(TARGETS maestro_py
                RUNTIME DESTINATION maestro
                LIBRARY DESTINATION maestro
                ARCHIVE DESTINATION maestro
            )
        else()
            install(TARGETS maestro_py LIBRARY DESTINATION maestro)
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/maestro/sinter.py")
            install(FILES
                maestro/__init__.py
                maestro/sinter.py
                DESTINATION maestro
            )
        endif()
    else()
        message(WARNING "Python 3.10+ (with Interpreter and Development.Module) not found. Skipping Python bindings.")
    endif()
endif()

# ==============================================================================
# 7. Test Suite
# ==============================================================================

if(COMPILE_TESTS)
    enable_testing()

    # Focused CPU tensor tests use baseline compiler ISA settings. They also
    # build without OpenMP and do not require the Aer/GPU simulator backends.
    add_executable(tensor_contraction_tests tests/TensorContractionTests.cpp)
    target_compile_definitions(tensor_contraction_tests PRIVATE
        MAESTRO_TENSOR_TEST_MAIN BOOST_ALL_NO_LIB NOMINMAX)
    if(NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(tensor_contraction_tests PRIVATE BOOST_SERIALIZATION_DYN_LINK)
    endif()
    target_link_libraries(tensor_contraction_tests PRIVATE Boost::serialization)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tensor_contraction_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME tensor_contractions COMMAND tensor_contraction_tests)
    if(WIN32 AND NOT Boost_USE_STATIC_LIBS)
        string(REPLACE ";" "\\;" tensor_test_path "$<TARGET_FILE_DIR:Boost::serialization>;$ENV{PATH}")
        set_tests_properties(tensor_contractions PROPERTIES ENVIRONMENT "PATH=${tensor_test_path}")
    endif()

    set(TESTSSRC
        tests/AerTests.cpp
        tests/DensityMatrixTests.cpp
        tests/CircTests.cpp
        tests/CliffordTests.cpp
        tests/CompSimTests.cpp
        tests/ExpVals.cpp
        tests/GpuTests.cpp
        tests/GpuDensityMatrixTests.cpp
        tests/GpuMPOTests.cpp
        tests/GpuVsCpuDensityMatrixTests.cpp
        tests/GpuVsCpuMPOTests.cpp
        tests/GpuVsCpuMPSTests.cpp
        tests/InitializeToBasisStateTests.cpp
        tests/MPSSimTests.cpp
        tests/Qasm.cpp
        tests/QasmCore.cpp
        tests/QasmDialects.cpp
        tests/QasmExpressions.cpp
        tests/QasmGateContract.cpp
        tests/QasmIntegration.cpp
        tests/QasmInputs.cpp
        tests/QasmModifiers.cpp
        tests/QasmSyntax.cpp
        tests/QasmValidation.cpp
        tests/QCSimTests.cpp
        tests/QCSimDensityMatrixTests.cpp
        tests/QCSimExtendedStabilizerTests.cpp
        tests/QCSimMPOTests.cpp
        tests/TensorsTests.cpp
        tests/PauliPropTests.cpp
        tests/ExtStabTests.cpp
        tests/MPSSwapsOptTests.cpp
        tests/NoiseModelTests.cpp
        tests/PathIntegralTests.cpp
        tests/NetwSimTests.cpp
        tests/FusionNetwTests.cpp
        tests/MPSNetwTests.cpp
        tests/Tests.cpp
    )

    set(ACTIVE_TESTSSRC ${TESTSSRC})
    if(NOT QAER)
        list(REMOVE_ITEM ACTIVE_TESTSSRC
            tests/AerTests.cpp
            tests/DensityMatrixTests.cpp
            tests/CompSimTests.cpp
            tests/CliffordTests.cpp
            tests/CircTests.cpp
            tests/ExtStabTests.cpp
            tests/PathIntegralTests.cpp
        )
    endif()

    add_executable(tests ${ACTIVE_TESTSSRC})
    target_link_libraries(tests PRIVATE Boost::serialization)
    if(MSVC AND NOT Boost_USE_STATIC_LIBS)
        # Tests.cpp supplies the Boost.Test implementation. Import only the
        # serialization DLL, not the test framework, in the other test files.
        get_target_property(tests_link_libraries tests LINK_LIBRARIES)
        list(REMOVE_ITEM tests_link_libraries Boost::dynamic_linking)
        set_property(TARGET tests PROPERTY LINK_LIBRARIES "${tests_link_libraries}")
        target_compile_definitions(tests PRIVATE BOOST_SERIALIZATION_DYN_LINK)
    endif()
    if(MAESTRO_ENABLE_COMPOSER)
        # This executable compiles Maestro sources without linking libmaestro.
        target_link_libraries(tests PRIVATE maestro_composer)
    endif()

    if(UNIX)
        target_link_libraries(tests PRIVATE dl)
    endif()

    if(SIMD_FLAGS)
        target_compile_options(tests PUBLIC ${SIMD_FLAGS})
    endif()

    if(QAER AND BLAS_FOUND)
        target_link_libraries(tests PUBLIC ${MAESTRO_LINALG_LIBRARIES})
    endif()

    if(OpenMP_CXX_FOUND)
        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(tests PUBLIC -Xpreprocessor -fopenmp)
            if(LIBOMP_PREFIX)
                target_include_directories(tests PUBLIC ${OpenMP_CXX_INCLUDE_DIRS} "${LIBOMP_PREFIX}/include")
                target_link_directories(tests PUBLIC "${LIBOMP_PREFIX}/lib")
            else()
                target_include_directories(tests PUBLIC ${OpenMP_CXX_INCLUDE_DIRS} "$ENV{OpenMP_ROOT}/include")
                target_link_directories(tests PUBLIC "$ENV{OpenMP_ROOT}/lib")
            endif()
            target_link_libraries(tests PUBLIC ${OpenMP_CXX_LIBRARIES})
        else()
            target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
            if(NOT MSVC)
                target_link_libraries(tests PRIVATE ${OpenMP_CXX_FLAGS})
            endif()
            target_compile_options(tests PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    endif()

    # Unity test: sampling_order_tests
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/SamplingOrderTests.cpp")
        if(CMAKE_VERSION VERSION_LESS "3.16")
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/SamplingOrderUnity.cpp"
                "#include \"${CMAKE_CURRENT_SOURCE_DIR}/Simulators/Core/Factory.cpp\"\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/tests/SamplingOrderTests.cpp\"\n")
            add_executable(sampling_order_tests "${CMAKE_CURRENT_BINARY_DIR}/SamplingOrderUnity.cpp")
        else()
            add_executable(sampling_order_tests Simulators/Core/Factory.cpp tests/SamplingOrderTests.cpp)
            set_target_properties(sampling_order_tests PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 0)
        endif()
        target_link_libraries(sampling_order_tests PRIVATE Boost::serialization ${CMAKE_DL_LIBS}
            $<TARGET_PROPERTY:maestro,INTERFACE_LINK_LIBRARIES>)
        target_compile_options(sampling_order_tests PRIVATE $<TARGET_PROPERTY:maestro,INTERFACE_COMPILE_OPTIONS>)
        target_include_directories(sampling_order_tests PRIVATE $<TARGET_PROPERTY:maestro,INTERFACE_INCLUDE_DIRECTORIES>)
        set_property(TARGET sampling_order_tests PROPERTY LINK_DIRECTORIES $<TARGET_PROPERTY:maestro,INTERFACE_LINK_DIRECTORIES>)

        add_test(NAME sampling_order COMMAND sampling_order_tests)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(NAME sampling_order_gpu COMMAND sampling_order_tests --gpu)
            set_tests_properties(sampling_order_gpu PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "gpu;hardware")
        endif()
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(distributed_gpu_error_tests tests/distributed_gpu/ErrorContractTests.cpp)
        target_link_libraries(distributed_gpu_error_tests PRIVATE ${CMAKE_DL_LIBS})
        add_test(NAME distributed_gpu_error_contract COMMAND distributed_gpu_error_tests)
        add_executable(distributed_gpu_tests tests/distributed_gpu/IntegrationTests.cpp)
        target_link_libraries(distributed_gpu_tests PRIVATE maestro)
        # Match Python's visibility boundary when checking singleton identity.
        set_target_properties(distributed_gpu_tests PROPERTIES CXX_VISIBILITY_PRESET hidden)
        add_executable(distributed_network_tests tests/distributed_gpu/NetworkMappingTests.cpp)
        target_link_libraries(distributed_network_tests PRIVATE maestro)
        set_property(TARGET distributed_network_tests APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--disable-new-dtags")
        add_test(NAME distributed_network_mapping COMMAND distributed_network_tests)
        set_property(TARGET distributed_gpu_tests APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--disable-new-dtags")
        add_test(NAME distributed_gpu_configuration COMMAND distributed_gpu_tests --config-only)
        add_test(NAME distributed_gpu_missing_probe COMMAND distributed_gpu_tests --unavailable-probe)
        set_tests_properties(distributed_gpu_missing_probe PROPERTIES
            ENVIRONMENT "MAESTRO_DIST_GPU_LIBRARY=/maestro-test-missing/plugin.so")
        add_test(NAME distributed_gpu_incompatible_probe COMMAND distributed_gpu_tests --unavailable-probe)
        set_tests_properties(distributed_gpu_incompatible_probe PROPERTIES
            ENVIRONMENT "MAESTRO_DIST_GPU_LIBRARY=libc.so.6")
        add_test(NAME distributed_gpu_missing_mpi_plugin COMMAND distributed_gpu_tests --missing-mpi-plugin)
        set_tests_properties(distributed_gpu_missing_mpi_plugin PROPERTIES
            ENVIRONMENT "MAESTRO_DIST_MPI_GPU_LIBRARY=/maestro-test-missing/libmaestro_gpu_distributed_mpi.so")
        add_test(NAME distributed_gpu_single COMMAND distributed_gpu_tests)
        add_test(NAME distributed_gpu_shared COMMAND distributed_gpu_tests --shared)
        add_test(NAME distributed_gpu_two_devices COMMAND distributed_gpu_tests --two-gpus)
        set_tests_properties(distributed_gpu_single distributed_gpu_shared distributed_gpu_two_devices
            PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;hardware;distributed" RUN_SERIAL TRUE TIMEOUT 180)
        if(MAESTRO_BUILD_MPI_GPU_TESTS)
            find_package(MPI REQUIRED COMPONENTS CXX)
            target_compile_definitions(distributed_gpu_tests PRIVATE MAESTRO_MPI_GPU_TESTS)
            target_link_libraries(distributed_gpu_tests PRIVATE MPI::MPI_CXX)
            add_test(NAME distributed_gpu_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:distributed_gpu_tests> --mpi ${MPIEXEC_POSTFLAGS})
            add_test(NAME distributed_gpu_mpi_seeds COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:distributed_gpu_tests> --mpi-seeds-only ${MPIEXEC_POSTFLAGS})
            set_tests_properties(distributed_gpu_mpi distributed_gpu_mpi_seeds PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "gpu;hardware;distributed;mpi" RUN_SERIAL TRUE TIMEOUT 240)
        endif()
    endif()

    # Linux-only GPU mock plugin and device tests
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(Threads REQUIRED)

        add_library(mock_gpu_plugin SHARED tests/gpu_device/MockPlugin.cpp)
        set_target_properties(mock_gpu_plugin PROPERTIES
            OUTPUT_NAME maestro_gpu_simulators
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mock_gpu")

        add_library(mock_distributed_gpu_plugin SHARED tests/distributed_gpu/MockPlugin.cpp)
        add_dependencies(distributed_gpu_tests mock_distributed_gpu_plugin)
        add_dependencies(distributed_network_tests mock_distributed_gpu_plugin)
        set_tests_properties(distributed_gpu_configuration distributed_network_mapping PROPERTIES
            ENVIRONMENT "MAESTRO_DIST_GPU_LIBRARY=$<TARGET_FILE:mock_distributed_gpu_plugin>")
        add_executable(gpu_licensing_tests tests/gpu_device/LicensingTests.cpp)
        target_link_libraries(gpu_licensing_tests PRIVATE maestro)
        set_property(TARGET gpu_licensing_tests APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--disable-new-dtags")
        add_dependencies(gpu_licensing_tests mock_gpu_plugin mock_distributed_gpu_plugin)
        foreach(backend gpu distributed mpi)
            foreach(outcome valid load init)
                add_test(NAME licensing_${backend}_${outcome} COMMAND ${CMAKE_COMMAND} -E env
                    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:mock_gpu_plugin>:$<TARGET_FILE_DIR:maestro>:$ENV{LD_LIBRARY_PATH}"
                    "MAESTRO_DIST_GPU_LIBRARY=$<TARGET_FILE:mock_distributed_gpu_plugin>"
                    "MAESTRO_DIST_MPI_GPU_LIBRARY=$<TARGET_FILE:mock_distributed_gpu_plugin>"
                    $<TARGET_FILE:gpu_licensing_tests> ${backend} ${outcome})
            endforeach()
        endforeach()

        add_executable(gpu_registry_tests tests/gpu_device/RegistryTests.cpp)
        target_link_libraries(gpu_registry_tests PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
        add_test(NAME gpu_registry COMMAND gpu_registry_tests $<TARGET_FILE:mock_gpu_plugin>)

        add_executable(gpu_multi_device_tests tests/gpu_device/MultiGpuTests.cpp)
        target_link_libraries(gpu_multi_device_tests PRIVATE maestro)
        set_property(TARGET gpu_multi_device_tests APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--disable-new-dtags")
        add_test(NAME gpu_multi_device COMMAND gpu_multi_device_tests)
        set_tests_properties(gpu_multi_device PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "gpu;hardware")

        add_executable(gpu_device_tests tests/gpu_device/SimulatorTests.cpp)
        target_link_libraries(gpu_device_tests PRIVATE maestro Threads::Threads)
        add_dependencies(gpu_device_tests mock_gpu_plugin)
        add_test(NAME gpu_device_configuration
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:mock_gpu_plugin>:$<TARGET_FILE_DIR:maestro>:$ENV{LD_LIBRARY_PATH}"
                $<TARGET_FILE:gpu_device_tests>)

        add_executable(gpu_svd_tests tests/gpu_device/SvdTests.cpp)
        target_link_libraries(gpu_svd_tests PRIVATE maestro Threads::Threads)
        add_dependencies(gpu_svd_tests mock_gpu_plugin)
        add_test(NAME gpu_svd_configuration
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:mock_gpu_plugin>:$<TARGET_FILE_DIR:maestro>:$ENV{LD_LIBRARY_PATH}"
                $<TARGET_FILE:gpu_svd_tests> --config-only)

        foreach(gpu_test gpu_device_tests gpu_svd_tests)
            set_property(TARGET ${gpu_test} APPEND_STRING PROPERTY
                LINK_FLAGS " -Wl,--disable-new-dtags")
        endforeach()
    endif()
endif()

# ==============================================================================
# 8. Installation & Documentation
# ==============================================================================

if(SKBUILD)
    install(TARGETS maestro DESTINATION maestro)
else()
    if(WIN32)
        install(TARGETS maestro
            RUNTIME DESTINATION .
            LIBRARY DESTINATION .
            ARCHIVE DESTINATION .
        )
        install(TARGETS maestroexe
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION bin
            ARCHIVE DESTINATION bin
        )
    else()
        install(TARGETS maestro DESTINATION .)
        install(TARGETS maestroexe DESTINATION bin)
    endif()
endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else()
    add_custom_target(doc
        COMMAND ${CMAKE_COMMAND} -E echo "Error: Doxygen not found. Please install Doxygen to generate documentation."
        COMMAND ${CMAKE_COMMAND} -E false
        COMMENT "Check for Doxygen"
        VERBATIM)
endif()

# Focused public native-API tests can be built without the full simulator suite.
option(MAESTRO_BUILD_REQUEST_TESTS "Build the native JSON request contract tests" OFF)
if(MAESTRO_BUILD_REQUEST_TESTS)
    enable_testing()
    add_executable(expectation_batch_tests tests/ExpectationBatchTests.cpp)
    target_include_directories(expectation_batch_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(expectation_batch_tests PRIVATE maestro Boost::json)
    add_test(NAME expectation_batches COMMAND expectation_batch_tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME gpu_tensor_queries_double COMMAND expectation_batch_tests --gpu)
        add_test(NAME gpu_tensor_queries_single COMMAND expectation_batch_tests --gpu --single)
        set_tests_properties(gpu_tensor_queries_double gpu_tensor_queries_single PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "gpu;hardware" RUN_SERIAL TRUE TIMEOUT 180)
        add_executable(tensor_dispatch_tests tests/gpu_device/TensorDispatchTests.cpp)
        target_link_libraries(tensor_dispatch_tests PRIVATE maestro)
        foreach(variant IN ITEMS current legacy)
            add_library(tensor_mock_${variant} SHARED tests/gpu_device/TensorMockPlugin.cpp)
            set_target_properties(tensor_mock_${variant} PROPERTIES
                OUTPUT_NAME maestro_gpu_simulators
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tensor_mock_${variant}")
            if(variant STREQUAL "legacy")
                target_compile_definitions(tensor_mock_${variant} PRIVATE MAESTRO_OLD_TENSOR_PLUGIN)
                add_test(NAME tensor_dispatch_${variant} COMMAND tensor_dispatch_tests --legacy)
            else()
                add_test(NAME tensor_dispatch_${variant} COMMAND tensor_dispatch_tests)
            endif()
            add_dependencies(tensor_dispatch_tests tensor_mock_${variant})
            set_tests_properties(tensor_dispatch_${variant} PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tensor_mock_${variant}:${CMAKE_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")
        endforeach()
    endif()
    add_executable(maestro_request_tests tests/RequestApi.cpp tests/RequestNoise.cpp tests/RequestOptions.cpp tests/RequestHeader.c)
    target_include_directories(maestro_request_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(maestro_request_tests PRIVATE maestro Boost::json)
    add_test(NAME maestro_request_api COMMAND maestro_request_tests)
    add_executable(fusion_interface_wrapper_tests tests/FusionInterfaceWrappers.cpp)
    target_link_libraries(fusion_interface_wrapper_tests PRIVATE maestro ${CMAKE_DL_LIBS})
    add_test(NAME fusion_interface_wrappers COMMAND fusion_interface_wrapper_tests $<TARGET_FILE:maestro>)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME gpu_tensor_cpp_wrappers COMMAND fusion_interface_wrapper_tests $<TARGET_FILE:maestro> --gpu)
        set_tests_properties(gpu_tensor_cpp_wrappers PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "gpu;hardware" RUN_SERIAL TRUE TIMEOUT 180)
        # These tests both link Maestro and load its explicit build path. Loading
        # an installed copy via LD_LIBRARY_PATH as well can corrupt shared state.
        # Preserve the caller's plugin search paths while selecting one Maestro.
        if(CMAKE_VERSION VERSION_LESS "3.22")
            set_tests_properties(fusion_interface_wrappers gpu_tensor_cpp_wrappers PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:maestro>:$ENV{LD_LIBRARY_PATH}")
        else()
            set_tests_properties(fusion_interface_wrappers gpu_tensor_cpp_wrappers PROPERTIES
                ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:maestro>")
        endif()
    endif()
    add_executable(maestro_request_abi_consumer tests/RequestHeader.c)
    target_compile_definitions(maestro_request_abi_consumer PRIVATE MAESTRO_ABI_CONSUMER_MAIN)
    target_link_libraries(maestro_request_abi_consumer PRIVATE maestro)
    add_test(NAME maestro_request_abi COMMAND maestro_request_abi_consumer)
endif()

# Worker and noise regressions also run in builds containing only request tests.
if(COMPILE_TESTS OR MAESTRO_BUILD_REQUEST_TESTS)
    enable_testing()
    add_executable(maestro_composer_integration_tests tests/ComposerIntegrationTests.cpp)
    target_compile_definitions(maestro_composer_integration_tests PRIVATE
        MAESTRO_EXPECT_COMPOSER=$<BOOL:${MAESTRO_ENABLE_COMPOSER}>)
    target_link_libraries(maestro_composer_integration_tests PRIVATE maestro)
    add_test(NAME maestro_composer_integration COMMAND maestro_composer_integration_tests)
    add_executable(network_job_tests tests/NetworkJobTests.cpp)
    target_link_libraries(network_job_tests PRIVATE maestro Boost::json)
    add_test(NAME network_jobs COMMAND network_job_tests)
    set_tests_properties(network_jobs PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
endif()

# Standalone native JSON runner; MPI is linked only into its optional sibling.
add_executable(maestro_request_worker maestroexe/RequestWorker.cpp)
set_target_properties(maestro_request_worker PROPERTIES OUTPUT_NAME maestro-request)
target_link_libraries(maestro_request_worker PRIVATE maestro Boost::json)
if(APPLE)
    set_target_properties(maestro_request_worker PROPERTIES
        INSTALL_RPATH "@loader_path/..;@loader_path/../lib"
    )
elseif(UNIX)
    set_target_properties(maestro_request_worker PROPERTIES
        INSTALL_RPATH "$ORIGIN/..:$ORIGIN/../lib"
    )
endif()

option(MAESTRO_BUILD_MPI_WORKER "Build the native collective MPI request worker" OFF)
if(MAESTRO_BUILD_MPI_WORKER)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_executable(maestro_mpi_worker maestroexe/RequestWorker.cpp)
    set_target_properties(maestro_mpi_worker PROPERTIES OUTPUT_NAME maestro-mpi-worker)
    target_compile_definitions(maestro_mpi_worker PRIVATE MAESTRO_REQUEST_MPI)
    target_link_libraries(maestro_mpi_worker PRIVATE maestro Boost::json MPI::MPI_CXX)
    if(APPLE)
        set_target_properties(maestro_mpi_worker PROPERTIES
            INSTALL_RPATH "@loader_path/..;@loader_path/../lib"
        )
    elseif(UNIX)
        set_target_properties(maestro_mpi_worker PROPERTIES
            INSTALL_RPATH "$ORIGIN/..:$ORIGIN/../lib"
        )
    endif()
    if(NOT SKBUILD)
        install(TARGETS maestro_mpi_worker RUNTIME DESTINATION bin)
    endif()
endif()
if(NOT SKBUILD)
    install(TARGETS maestro_request_worker RUNTIME DESTINATION bin)
endif()

if(MAESTRO_BUILD_REQUEST_TESTS)
    add_test(NAME maestro_request_cli COMMAND ${CMAKE_COMMAND}
        -DWORKER=$<TARGET_FILE:maestro_request_worker> -DMODE=success
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/request_worker.cmake)
    if(MAESTRO_BUILD_MPI_WORKER)
        add_executable(maestro_request_mpi_seed_tests tests/RequestMpiSeeds.cpp)
        target_link_libraries(maestro_request_mpi_seed_tests PRIVATE maestro Boost::json MPI::MPI_CXX)
        foreach(mode success failure)
            add_test(NAME maestro_request_mpi_${mode} COMMAND ${CMAKE_COMMAND}
                -DWORKER=$<TARGET_FILE:maestro_mpi_worker> -DMODE=${mode}
                -DMPI_LAUNCHER=${MPIEXEC_EXECUTABLE} -DMPI_COUNT_FLAG=${MPIEXEC_NUMPROC_FLAG}
                "-DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/request_worker.cmake)
            set_tests_properties(maestro_request_mpi_${mode} PROPERTIES PROCESSORS 2 TIMEOUT 40)
        endforeach()
    endif()
    option(MAESTRO_RUN_GPU_REQUEST_TESTS "Run GPU request tests (requires plugin, license, CUDA device)" OFF)
    if(MAESTRO_RUN_GPU_REQUEST_TESTS)
        add_test(NAME maestro_request_gpu COMMAND ${CMAKE_COMMAND}
            -DWORKER=$<TARGET_FILE:maestro_request_worker> -DMODE=gpu
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/request_worker.cmake)
        set_tests_properties(maestro_request_gpu PROPERTIES TIMEOUT 40)
    endif()
endif()


option(MAESTRO_RUN_MPI_GPU_REQUEST_TESTS "Run native MPI GPU request tests (requires rank-local devices, plugins, licensing and CUDA-aware MPI)" OFF)
if(MAESTRO_RUN_MPI_GPU_REQUEST_TESTS)
    if(NOT MAESTRO_BUILD_REQUEST_TESTS OR NOT MAESTRO_BUILD_MPI_WORKER)
        message(FATAL_ERROR "MPI GPU request tests require MAESTRO_BUILD_REQUEST_TESTS and MAESTRO_BUILD_MPI_WORKER")
    endif()
    add_test(NAME maestro_request_mpi_gpu_seeds COMMAND ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:maestro_request_mpi_seed_tests> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(maestro_request_mpi_gpu_seeds PROPERTIES
        LABELS "mpi;gpu;hardware;request" PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 180)
    foreach(mode mpi_gpu mpi_gpu_noise mpi_gpu_failure)
        add_test(NAME maestro_request_${mode} COMMAND ${CMAKE_COMMAND}
            -DWORKER=$<TARGET_FILE:maestro_mpi_worker> -DMODE=${mode}
            -DMPI_LAUNCHER=${MPIEXEC_EXECUTABLE} -DMPI_COUNT_FLAG=${MPIEXEC_NUMPROC_FLAG}
            "-DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/request_worker.cmake)
        set_tests_properties(maestro_request_${mode} PROPERTIES
            LABELS "mpi;gpu;hardware;request" PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60)
    endforeach()
endif()

if(COMPILE_TESTS)
    find_package(Threads REQUIRED)
    add_executable(backend_specialization_tests tests/BackendSpecializationTests.cpp)
    target_include_directories(backend_specialization_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(backend_specialization_tests PRIVATE maestro Threads::Threads)
    add_test(NAME backend_specialization COMMAND backend_specialization_tests)
    add_executable(gate_fusion_tests tests/GateFusionTests.cpp)
    target_include_directories(gate_fusion_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gate_fusion_tests PRIVATE maestro)
    add_test(NAME gate_fusion_tests COMMAND gate_fusion_tests)
    add_executable(fusion_routing_tests tests/FusionRoutingTests.cpp)
    target_include_directories(fusion_routing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(fusion_routing_tests PRIVATE maestro)
    add_test(NAME fusion_routing_tests COMMAND fusion_routing_tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME gate_fusion_gpu COMMAND gate_fusion_tests --gpu)
        add_test(NAME gate_fusion_distributed COMMAND gate_fusion_tests --distributed)
        set_tests_properties(gate_fusion_gpu gate_fusion_distributed PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "gpu;hardware" RUN_SERIAL TRUE TIMEOUT 180)
    endif()
    add_executable(gate_fusion_benchmark EXCLUDE_FROM_ALL tests/GateFusionBenchmark.cpp)
    target_include_directories(gate_fusion_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gate_fusion_benchmark PRIVATE maestro)
endif()

if(COMPILE_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(gpu_mpo_api_tests tests/GpuMPOApiTests.cpp)
    target_link_libraries(gpu_mpo_api_tests PRIVATE maestro Boost::json)
    add_test(NAME cpu_mpo_api COMMAND gpu_mpo_api_tests --cpu)
    add_test(NAME gpu_mpo_api COMMAND gpu_mpo_api_tests)
    set_tests_properties(gpu_mpo_api PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "gpu;hardware" RUN_SERIAL TRUE TIMEOUT 180)
endif()

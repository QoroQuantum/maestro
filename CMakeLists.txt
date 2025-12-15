cmake_minimum_required(VERSION 3.10)
project(maestro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Qiskit Aer is optional - disabled by default for easier builds
# Enable only if AER_INCLUDE_DIR is explicitly set
set(QAER False)

# Check if explicitly disabled
if (NOT "$ENV{NO_QISKIT_AER}" STREQUAL "")
    add_definitions(-DNO_QISKIT_AER)
    set(QAER False)
    message(STATUS "Qiskit Aer support explicitly disabled via NO_QISKIT_AER")
# Check if AER_INCLUDE_DIR is set (enables Aer)
elseif (NOT "$ENV{AER_INCLUDE_DIR}" STREQUAL "")
    set(QAER True)
    message(STATUS "Qiskit Aer support enabled (AER_INCLUDE_DIR is set)")
# Default: disabled (user-friendly for PyPI builds)
else()
    add_definitions(-DNO_QISKIT_AER)
    set(QAER False)
    message(STATUS "Qiskit Aer support disabled by default. Set AER_INCLUDE_DIR to enable.")
endif()

# Include FetchContent for automatic dependency management
include(FetchContent)

# Helper function to find FFTW3
function(find_fftw3)
    if (NOT DEFINED FFTW3_DIR)
        SET( FFTW3_DIR "$ENV{FFTW3_DIR}" PARENT_SCOPE)
        IF( NOT FFTW3_DIR )
            # Try to find FFTW3 using pkg-config
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(FFTW3 QUIET fftw3)
                if(FFTW3_FOUND)
                    get_filename_component(FFTW3_DIR "${FFTW3_INCLUDE_DIRS}" DIRECTORY)
                    set(FFTW3_DIR ${FFTW3_DIR} PARENT_SCOPE)
                endif()
            endif()
            
            # If still not found, try common locations
            # Note: QCSim expects FFTW3_DIR to point to the include directory, not the root
            if( NOT FFTW3_DIR )
                if(EXISTS "/usr/include/fftw3.h")
                    set(FFTW3_DIR "/usr/include" PARENT_SCOPE)
                elseif(EXISTS "/usr/local/include/fftw3.h")
                    set(FFTW3_DIR "/usr/local/include" PARENT_SCOPE)
                elseif(EXISTS "/opt/homebrew/include/fftw3.h")
                    set(FFTW3_DIR "/opt/homebrew/include" PARENT_SCOPE)
                endif()
            endif()
            
            if( NOT FFTW3_DIR )
                MESSAGE( WARNING "FFTW3 not found. QCSim requires FFTW3. Please install libfftw3-dev (Ubuntu/Debian) or fftw (macOS) and set FFTW3_DIR environment variable if needed." )
            else()
                message(STATUS "Found FFTW3 at: ${FFTW3_DIR}")
            endif()
        ENDIF()
    endif()
endfunction()

# Eigen 5 (header-only, can be fetched automatically)
if (NOT DEFINED EIGEN5_INCLUDE_DIR)
    SET( EIGEN5_INCLUDE_DIR "$ENV{EIGEN5_INCLUDE_DIR}" )
    IF( NOT EIGEN5_INCLUDE_DIR )
        # Auto-fetch Eigen if not provided
        message(STATUS "EIGEN5_INCLUDE_DIR not set, fetching Eigen 5.0.0...")
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(eigen)
        set(EIGEN5_INCLUDE_DIR ${eigen_SOURCE_DIR})
        message(STATUS "Eigen fetched to: ${EIGEN5_INCLUDE_DIR}")
    ENDIF()
ENDIF()

# QCSim (can be fetched automatically, but requires FFTW3)
if (NOT DEFINED QCSIM_INCLUDE_DIR)
    SET( QCSIM_INCLUDE_DIR "$ENV{QCSIM_INCLUDE_DIR}" )
    IF( NOT QCSIM_INCLUDE_DIR )
        # QCSim requires FFTW3 - find it first
        find_fftw3()
        
        # Set FFTW3_DIR and library paths for QCSim's CMake
        if( FFTW3_DIR )
            set(ENV{FFTW3_DIR} ${FFTW3_DIR})
            # QCSim may need the library directory path
            # Try to find the actual library location
            if(EXISTS "${FFTW3_DIR}/lib/x86_64-linux-gnu/libfftw3.a")
                set(ENV{FFTW3_LIB_DIR} "${FFTW3_DIR}/lib/x86_64-linux-gnu")
                # Also set CMAKE variables that QCSim might use
                set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FFTW3_DIR}" CACHE STRING "" FORCE)
                link_directories("${FFTW3_DIR}/lib/x86_64-linux-gnu")
            elseif(EXISTS "${FFTW3_DIR}/lib/libfftw3.a")
                set(ENV{FFTW3_LIB_DIR} "${FFTW3_DIR}/lib")
                set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FFTW3_DIR}" CACHE STRING "" FORCE)
                link_directories("${FFTW3_DIR}/lib")
            elseif(EXISTS "${FFTW3_DIR}/lib64/libfftw3.a")
                set(ENV{FFTW3_LIB_DIR} "${FFTW3_DIR}/lib64")
                set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FFTW3_DIR}" CACHE STRING "" FORCE)
                link_directories("${FFTW3_DIR}/lib64")
            endif()
        endif()
        
        # Auto-fetch QCSim if not provided
        message(STATUS "QCSIM_INCLUDE_DIR not set, fetching QCSim...")
        FetchContent_Declare(
            qcsim
            GIT_REPOSITORY https://github.com/aromanro/QCSim.git
            GIT_TAG master
        )
        
        # Just populate the source (don't build it with add_subdirectory)
        FetchContent_GetProperties(qcsim)
        if(NOT qcsim_POPULATED)
            FetchContent_Populate(qcsim)
        endif()
        
        set(QCSIM_INCLUDE_DIR ${qcsim_SOURCE_DIR}/QCSim)
        message(STATUS "QCSim fetched to: ${QCSIM_INCLUDE_DIR}")
    ENDIF()
ENDIF()

if (QAER)
    # nlohmann/json (header-only, can be fetched automatically)
    if (NOT DEFINED JSON_INCLUDE_DIR)
        SET( JSON_INCLUDE_DIR "$ENV{JSON_INCLUDE_DIR}" )
        IF( NOT JSON_INCLUDE_DIR )
            # Auto-fetch nlohmann/json if not provided
            message(STATUS "JSON_INCLUDE_DIR not set, fetching nlohmann/json...")
            FetchContent_Declare(
                json
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG v3.11.2
            )
            FetchContent_MakeAvailable(json)
            set(JSON_INCLUDE_DIR ${json_SOURCE_DIR}/single_include)
            message(STATUS "nlohmann/json fetched to: ${JSON_INCLUDE_DIR}")
        ENDIF()
    ENDIF()

    # Qiskit Aer (requires AER_INCLUDE_DIR to be set)
    if (NOT DEFINED AER_INCLUDE_DIR)
        SET( AER_INCLUDE_DIR "$ENV{AER_INCLUDE_DIR}" )
        IF( NOT AER_INCLUDE_DIR )
            # If we reach here, QAER is True but AER_INCLUDE_DIR is not set
            # This shouldn't happen with the new logic, but handle it gracefully
            MESSAGE( WARNING "AER_INCLUDE_DIR not set but Aer support is enabled. Disabling Aer support." )
            add_definitions(-DNO_QISKIT_AER)
            set(QAER False)
        ENDIF()
    ENDIF()
endif()

# add here common sources

set(MAESTROSRC maestrolib/Interface.cpp)
if(WIN32)
    list(APPEND MAESTROSRC maestrolib/dllmain.cpp)
else()
    list(APPEND MAESTROSRC Simulators/Factory.cpp)
endif()

set(MAESTROEXESRC maestroexe/maestroexe.cpp)


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

find_package(OpenMP REQUIRED)

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

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
    if(UNIX OR APPLE)
        if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            message("Setting SIMD...")
            set(SIMD_FLAGS "-mfma;-mavx2")
            enable_cxx_flag("-mpopcnt")
        endif()
    elseif(MSVC)
        message("Setting SIMD...")
        set(SIMD_FLAGS "/arch:AVX2")
    endif()
endif()


# Boost can be provided via:
# 1. BOOST_ROOT environment variable (set by build.sh for local builds)
# 2. System installation (for PyPI builds)
if (NOT "$ENV{BOOST_ROOT}" STREQUAL "")
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    message(STATUS "Using Boost from BOOST_ROOT: ${BOOST_ROOT}")
endif()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_DEBUG_LIBS OFF)
find_package(Boost REQUIRED COMPONENTS program_options json container serialization)

IF(NOT Boost_FOUND)
    MESSAGE( FATAL_ERROR "\n"
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
ENDIF()

if (QAER)
    INCLUDE_DIRECTORIES ( ${EIGEN5_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QCSIM_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${AER_INCLUDE_DIR})

    find_package(BLAS REQUIRED)
    IF(NOT BLAS_FOUND)
        MESSAGE( FATAL_ERROR "Couldn't find blas libs")
    ENDIF()
else()
    INCLUDE_DIRECTORIES ( ${EIGEN5_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QCSIM_INCLUDE_DIR})
endif()

LINK_LIBRARIES()

add_executable(maestroexe ${MAESTROEXESRC})
add_library(maestro SHARED ${MAESTROSRC})

set_target_properties(maestro PROPERTIES OUTPUT_NAME "maestro")
set_target_properties(maestroexe PROPERTIES OUTPUT_NAME "maestro")

# Set rpath for library linking (helps with PyPI distribution)
if(UNIX AND NOT APPLE)
    set_target_properties(maestro PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

install(TARGETS maestro DESTINATION .)
install(TARGETS maestroexe DESTINATION bin)

if(OpenMP_CXX_FOUND)
    message("Setting OpenMP")
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_program(BREW_CMD brew)
        if(BREW_CMD)
            execute_process(COMMAND ${BREW_CMD} --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
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
        target_link_libraries(maestro PRIVATE ${OpenMP_CXX_FLAGS})
        target_compile_options(maestro PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endif()

IF(QAER AND BLAS_FOUND)
    message("Setting linking blas")
    target_link_libraries(maestro PUBLIC ${BLAS_LIBRARIES})
ENDIF()

IF(SIMD_FLAGS)
    message("Setting SIMD flags")
    target_compile_options(maestro PUBLIC ${SIMD_FLAGS})
ENDIF()

if (UNIX)
    target_link_libraries(maestro PRIVATE dl)
    target_link_libraries(maestroexe PRIVATE dl)    
endif()

IF(Boost_FOUND)
    target_link_libraries(maestro PRIVATE Boost::serialization)
    target_link_libraries(maestroexe PRIVATE Boost::serialization)
    target_link_libraries(maestroexe PRIVATE Boost::program_options)
ENDIF()

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

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    # Find Python 3.8 or higher (matches pyproject.toml requires-python = ">=3.8")
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python ${Python_VERSION} (${Python_EXECUTABLE})")
        message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
        message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")

        include(FetchContent)
        FetchContent_Declare(
          nanobind
          GIT_REPOSITORY https://github.com/wjakob/nanobind.git
          GIT_TAG        v2.0.0
        )
        FetchContent_MakeAvailable(nanobind)

        nanobind_add_module(maestro_py python/bindings.cpp)
        set_target_properties(maestro_py PROPERTIES OUTPUT_NAME "maestro")
        target_link_libraries(maestro_py PRIVATE maestro)
        target_include_directories(maestro_py PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/maestrolib
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

        # Set RPATH/RPATH-related properties
        if(APPLE)
            set_target_properties(maestro_py PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "@loader_path;@loader_path/../lib"
            )
            set_target_properties(maestro PROPERTIES
                INSTALL_NAME_DIR "@rpath"
            )
        elseif(UNIX AND NOT APPLE)
            set_target_properties(maestro_py PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()

        install(TARGETS maestro_py LIBRARY DESTINATION .)
    else()
        message(WARNING "Python 3.8+ (with Interpreter and Development.Module) not found. Skipping Python bindings.")
    endif()
endif()

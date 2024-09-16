# Inherit default options
include("${CMAKE_CURRENT_LIST_DIR}/default.cmake")
# Disable Z3.
set(USE_Z3 OFF CACHE BOOL "Disable Z3" FORCE)
# Enable fuzzers
set(OSSFUZZ ON CACHE BOOL "Enable fuzzer build" FORCE)
set(LIB_FUZZING_ENGINE $ENV{LIB_FUZZING_ENGINE} CACHE STRING "Use fuzzer back-end defined by environment variable" FORCE)
# Link statically against boost libraries
set(BOOST_FOUND ON CACHE BOOL "" FORCE)
set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Link against static Boost libraries" FORCE)
set(Boost_USE_STATIC_RUNTIME ON CACHE BOOL "Link against static Boost runtime library" FORCE)

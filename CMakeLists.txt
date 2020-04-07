set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)
include_directories(
    /opt/intel/mkl/include
    /usr/local/Cellar/openblas/0.3.9/include
)

link_directories(
    /opt/intel/mkl/lib
    /usr/local/Cellar/openblas/0.3.9/lib
)

add_executable(main_mkl main.cpp)

target_link_libraries(
    main_mkl PRIVATE mkl_rt
)

target_compile_definitions(
    main_mkl PRIVATE "USE_MKL"
)
add_executable(main_oblas main.cpp)

target_link_libraries(
    main_oblas PRIVATE openblas
)

target_compile_definitions(
    main_oblas PRIVATE "USE_OPENBLAS"
)
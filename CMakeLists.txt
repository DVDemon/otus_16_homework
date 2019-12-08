cmake_minimum_required(VERSION 3.2)

if($ENV{TRAVIS_BUILD_NUMBER})
 project(16_homework VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else()
 project(16_homework VERSION 0.0.3)
endif()

#SET(CMAKE_CXX_COMPILER /usr/local/Cellar/gcc/9.2.0_2/bin/c++-9)

find_package(Threads)


add_executable(rclst rclst.cpp)
add_executable(rclss rclss.cpp)

set_target_properties(rclst PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
set_target_properties(rclss PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

target_include_directories(rclst PRIVATE "${CMAKE_BINARY_DIR}")
target_include_directories(rclss PRIVATE "${CMAKE_BINARY_DIR}")

target_link_libraries(rclst PRIVATE ${CMAKE_THREAD_LIBS_INIT} dlib lapack blas)
target_link_libraries(rclss PRIVATE ${CMAKE_THREAD_LIBS_INIT} dlib lapack blas)

install(TARGETS rclst RUNTIME DESTINATION bin)
install(TARGETS rclss RUNTIME DESTINATION bin)


set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT ddzuba@yandex.ru)
include(CPack)


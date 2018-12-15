cmake_minimum_required(VERSION 2.8)
project( lab2 )
find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )
add_executable( lab2 lab2_pthread.cpp )
target_link_libraries( lab2 Threads::Threads)
target_link_libraries( lab2 ${OpenCV_LIBS} )
set(THREADS_PREFER_PTHREAD_FLAG ON)


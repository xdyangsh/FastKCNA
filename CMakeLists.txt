cmake_minimum_required (VERSION 2.6)
project (KGRAPH)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11 -DDEBUG -DNSG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

MESSAGE(STATUS ${CMAKE_CXX_FLAGS})

find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)
include_directories(${Boost_INCLUDE_DIRS})

set(LIB_SRC kgraph.cpp metric.cpp neighbor.h)
include_directories(${KGRAPH_SOURCE_DIR})
add_library(kgraph SHARED ${LIB_SRC})
add_library(kgrapha STATIC ${LIB_SRC})
set_target_properties(kgrapha PROPERTIES OUTPUT_NAME kgraph)

add_executable(fasthnsw fasthnsw.cpp)
add_executable(fastnsg fastnsg.cpp)
add_executable(fasttaumng fasttaumng.cpp)
add_executable(fastalphapg fastalphapg.cpp)
add_executable(fastnsw fastnsw.cpp)

TARGET_LINK_LIBRARIES(fasthnsw kgrapha rt ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(fastnsg kgrapha rt ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(fasttaumng kgrapha rt ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(fastalphapg kgrapha rt ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(fastnsw kgrapha rt ${Boost_LIBRARIES})

install (FILES kgraph.h kgraph-data.h DESTINATION include) 
install (TARGETS kgraph kgrapha
	 LIBRARY DESTINATION lib
	 ARCHIVE DESTINATION lib)

	 
	 

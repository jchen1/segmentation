cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
file( GLOB segmentation_SRC
      "src/*.h"
      "src/*.cpp"
)
add_executable( segmentation ${segmentation_SRC} )
add_definitions( -std=c++11 )
target_link_libraries( segmentation ${OpenCV_LIBS} )
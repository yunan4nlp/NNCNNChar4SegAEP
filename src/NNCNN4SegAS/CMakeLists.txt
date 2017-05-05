
include_directories(
  model
)

add_executable(NNCNN4SegAS NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegAS ${LIBS})

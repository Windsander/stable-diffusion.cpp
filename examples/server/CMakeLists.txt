set(TARGET server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_11)

# Add submodules
add_subdirectory(httplib)
#add_subdirectory(rapidjson)

# Link submodules to the target
target_link_libraries(${TARGET} PRIVATE httplib)

# Include rapidjson headers
target_include_directories(${TARGET} PRIVATE rapidjson/include httplib)
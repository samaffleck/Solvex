# Add a test executable
add_executable(test_solvex TestSolvex.cpp)
target_link_libraries(test_solvex PRIVATE Solvex)

# Register test with CTest
add_test(NAME TestSolvex COMMAND test_solvex)

# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test-philox PRIVATE m)
endif()

# test-vae: flow VAE decoder parity harness against the torch reference.
add_executable(test-vae test-vae.cpp)
link_ggml_backends(test-vae)

# test-cond: condition encoder parity harness against the torch reference.
add_executable(test-cond test-cond.cpp)
link_ggml_backends(test-cond)

# test-dit: flow matching DiT parity harness against the torch reference.
add_executable(test-dit test-dit.cpp)
link_ggml_backends(test-dit)

# test-depth: RVQ depth decoder parity harness against the torch reference.
add_executable(test-depth test-depth.cpp)
link_ggml_backends(test-depth)

# test-lm: global LM parity harness against the transformers reference.
add_executable(test-lm test-lm.cpp)
link_ggml_backends(test-lm)

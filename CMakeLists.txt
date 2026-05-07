cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# Server dependencies (header-only, fetched at configure time)
option(QWEN3_TTS_SERVER "Build the OpenAI-compatible TTS server" ON)
if(QWEN3_TTS_SERVER)
    include(FetchContent)
    FetchContent_Declare(httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG        v0.20.0
    )
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(httplib nlohmann_json)
endif()

# PIC is needed for the shared library — set globally so GGML libs get it too
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# GGML as subdirectory — options like GGML_VULKAN, GGML_METAL pass through automatically
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
add_subdirectory(${GGML_DIR})

# Qwen3-specific CUDA megakernel (Phase A specialized Q8_0 MMVQ + Phase B
# per-layer talker megakernel — see kobbler/docker/tts-qwen3-dev/HANDOFF-
# megakernel-v0.md). Built when GGML_CUDA is on; otherwise the install()
# hook is a no-op and tts_transformer omits the dependency.
if(GGML_CUDA)
    enable_language(CUDA)
    add_library(qwen3_megakernel STATIC
        src/cuda/qwen3_megakernel.cu
    )
    target_include_directories(qwen3_megakernel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    )
    target_include_directories(qwen3_megakernel PRIVATE
        ${GGML_DIR}/include
    )
    set_target_properties(qwen3_megakernel PROPERTIES
        CUDA_STANDARD          17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    if(CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(qwen3_megakernel PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
    target_link_libraries(qwen3_megakernel PUBLIC ggml)
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/tokenizer_unicode.cpp
    src/tokenizer_unicode_data.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(text_tokenizer PUBLIC
    ggml
    Threads::Threads
)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(tts_transformer PUBLIC
    ggml
    Threads::Threads
)
if(GGML_CUDA)
    target_link_libraries(tts_transformer PUBLIC qwen3_megakernel)
    target_compile_definitions(tts_transformer PUBLIC QWEN3_TTS_HAS_MEGAKERNEL=1)
endif()
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    ggml
    Threads::Threads
)

# Audio codec encoder library (Mimi encoder for ICL voice cloning)
add_library(audio_codec_encoder STATIC
    src/audio_codec_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_codec_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_codec_encoder PUBLIC
    ggml
    Threads::Threads
)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    ggml
    Threads::Threads
)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_codec_encoder
    audio_tokenizer_decoder
    Threads::Threads
)

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Streaming-vs-oneshot parity test for the audio decoder
add_executable(test_streaming_parity
    tests/test_streaming_parity.cpp
)
target_link_libraries(test_streaming_parity PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# ICL diagnostic dump
add_executable(test_icl_dump
    tests/test_icl_dump.cpp
)
target_link_libraries(test_icl_dump PRIVATE
    qwen3_tts
    Threads::Threads
)

# Server executable (OpenAI-compatible TTS API)
if(QWEN3_TTS_SERVER)
    # In-process audio encoders (mp3 via libmp3lame, ogg-opus via libopus +
    # ogg muxer) for the /v1/audio/speech response_format options. Linked
    # via pkg-config against the BtBN ffmpeg-7.1 shared-libs install
    # dropped into /usr/local by the builder image. lgpl variants of BtBN
    # don't ship libmp3lame — image must use the gpl-shared tarball.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavcodec
        libavformat
        libavutil
    )

    add_executable(qwen3-tts-server
        src/server.cpp
        src/audio/ffmpeg_encode.cpp
    )
    target_link_libraries(qwen3-tts-server PRIVATE
        qwen3_tts
        httplib::httplib
        nlohmann_json::nlohmann_json
        PkgConfig::LIBAV
    )
endif()

# Install targets
# collect installable targets
set(_install_targets text_tokenizer tts_transformer audio_tokenizer_encoder audio_codec_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli)
if(QWEN3_TTS_SERVER)
    list(APPEND _install_targets qwen3-tts-server)
endif()
install(TARGETS ${_install_targets}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_codec_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

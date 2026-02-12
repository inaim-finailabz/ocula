#
# Flutter Llama Podspec - iOS plugin configuration with llama.cpp
#
Pod::Spec.new do |s|
  s.name             = 'flutter_llama'
  s.version          = '1.0.0'
  s.summary          = 'Flutter plugin for LLM inference with llama.cpp and GGUF models'
  s.description      = <<-DESC
Flutter plugin for running LLM inference with llama.cpp and GGUF models on iOS.
Supports GPU acceleration via Metal and CPU optimization via Accelerate framework.
                       DESC
  s.homepage         = 'https://github.com/nativemind/flutter_llama'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'NativeMind' => 'licensing@nativemind.net' }
  s.source           = { :path => '.' }

  # Source files — plugin classes + mtmd multimodal via include wrapper (mtmd_sources.mm)
  s.source_files = 'Classes/**/*.{swift,h,m,mm}'
  s.public_header_files = 'Classes/**/*.h'

  # Use xcframework (supports device + simulator)
  s.vendored_frameworks = 'llama.xcframework'

  # Preserve llama.cpp headers and mtmd headers
  s.preserve_paths = '../llama.cpp/include/**/*', '../llama.cpp/ggml/include/**/*', '../llama.cpp/tools/mtmd/**/*', '../llama.cpp/vendor/**/*'

  # C++ settings
  s.library = 'c++'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'GCC_ENABLE_CPP_EXCEPTIONS' => 'YES',
    'GCC_ENABLE_CPP_RTTI' => 'YES',
    'CLANG_WARN_DOCUMENTATION_COMMENTS' => 'NO',
    'GCC_WARN_INHIBIT_ALL_WARNINGS' => 'YES',
    'HEADER_SEARCH_PATHS' => '"${PODS_TARGET_SRCROOT}/../llama.cpp/include" "${PODS_TARGET_SRCROOT}/../llama.cpp/ggml/include" "${PODS_TARGET_SRCROOT}/../llama.cpp/tools/mtmd" "${PODS_TARGET_SRCROOT}/../llama.cpp/tools/mtmd/models" "${PODS_TARGET_SRCROOT}/../llama.cpp" "${PODS_TARGET_SRCROOT}/../llama.cpp/vendor" "${PODS_TARGET_SRCROOT}/../llama.cpp/vendor/stb" "${PODS_TARGET_SRCROOT}/../llama.cpp/vendor/miniaudio"',
    'OTHER_LDFLAGS' => '$(inherited) -framework "llama"',
  }

  # Frameworks for GPU acceleration and optimization
  s.frameworks = 'Metal', 'MetalKit', 'MetalPerformanceShaders', 'Accelerate'

  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.swift_version = '5.0'
end

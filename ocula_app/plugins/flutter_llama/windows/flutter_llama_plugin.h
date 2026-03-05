#pragma once

#include <flutter/method_channel.h>
#include <flutter/event_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <flutter/event_stream_handler_functions.h>

#include <memory>
#include <mutex>
#include <string>

class FlutterLlamaPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  FlutterLlamaPlugin();
  virtual ~FlutterLlamaPlugin();

  FlutterLlamaPlugin(const FlutterLlamaPlugin&) = delete;
  FlutterLlamaPlugin& operator=(const FlutterLlamaPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // Text stream event sink (set when Dart listens on flutter_llama/stream)
  std::shared_ptr<flutter::EventSink<flutter::EncodableValue>> stream_sink_;
  std::mutex stream_sink_mutex_;

  void SendStreamToken(const std::string& token);
  void SendStreamEnd();
  void SendStreamError(const std::string& code, const std::string& message);
};

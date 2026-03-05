#include "flutter_llama_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "flutter_llama_plugin.h"

void FlutterLlamaPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  FlutterLlamaPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}

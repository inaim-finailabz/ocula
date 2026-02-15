package com.finailabz.ocula

import android.os.Bundle
import com.google.android.play.core.assetpacks.AssetPackLocation
import com.google.android.play.core.assetpacks.AssetPackManager
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
import com.google.android.play.core.assetpacks.AssetPackStates
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.finailabz.ocula/asset_pack"
    private lateinit var assetPackManager: AssetPackManager

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        assetPackManager = AssetPackManagerFactory.getInstance(this)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getAssetPackPath" -> {
                        val packName = call.argument<String>("packName") ?: "models_pack"
                        val fileName = call.argument<String>("fileName") ?: ""
                        val location: AssetPackLocation? =
                            assetPackManager.getPackLocations()[packName]
                        if (location != null) {
                            val path = "${location.assetsPath()}/$fileName"
                            val file = File(path)
                            if (file.exists()) {
                                result.success(path)
                            } else {
                                result.success(null)
                            }
                        } else {
                            result.success(null)
                        }
                    }
                    "isAssetPackAvailable" -> {
                        val packName = call.argument<String>("packName") ?: "models_pack"
                        val location = assetPackManager.getPackLocations()[packName]
                        result.success(location != null)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}

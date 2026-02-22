## Flutter wrapper
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.util.** { *; }
-keep class io.flutter.view.** { *; }
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }

## Play Asset Delivery
-keep class com.google.android.play.core.** { *; }

## Don't warn about missing Play Core classes referenced by Flutter
-dontwarn com.google.android.play.core.**
-dontwarn com.google.android.gms.common.annotation.NoNullnessRewrite

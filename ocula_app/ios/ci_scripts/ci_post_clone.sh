#!/bin/sh
set -e

REPO_PATH="${CI_PRIMARY_REPOSITORY_PATH:-$PWD}"
APP_PATH="$REPO_PATH/ocula_app"
[ -d "$APP_PATH" ] || APP_PATH="$REPO_PATH"

# ── Install Flutter ──────────────────────────────────────────────────
if ! command -v flutter >/dev/null 2>&1; then
  echo "[ci] Installing Flutter (stable)..."
  git clone https://github.com/flutter/flutter.git --depth 1 -b stable "$HOME/flutter"
  export PATH="$HOME/flutter/bin:$PATH"
fi

# Resolve the Flutter SDK root (needed for Generated.xcconfig)
FLUTTER_ROOT="$(cd "$(dirname "$(command -v flutter)")" && cd .. && pwd)"
echo "[ci] Flutter root: $FLUTTER_ROOT"
flutter --version

# ── Dart / pub dependencies ──────────────────────────────────────────
cd "$APP_PATH"
echo "[ci] flutter pub get..."
flutter pub get

# ── Generated.xcconfig ───────────────────────────────────────────────
# flutter pub get should create this, but on Xcode Cloud it sometimes
# doesn't (first-time Flutter clone, no Xcode context during pub).
# Generate it manually if missing so the archive step can load the project.
XCCONFIG="ios/Flutter/Generated.xcconfig"
if [ -f "$XCCONFIG" ]; then
  echo "[ci] Generated.xcconfig present"
else
  echo "[ci] Generated.xcconfig missing — generating from pubspec.yaml..."
  VERSION=$(grep 'version:' pubspec.yaml | head -1 | awk '{print $2}')
  BUILD_NAME=$(echo "$VERSION" | cut -d'+' -f1)
  BUILD_NUMBER=$(echo "$VERSION" | cut -d'+' -f2)
  mkdir -p "ios/Flutter"
  cat > "$XCCONFIG" << XCCONFIG
// This is a generated file; do not edit or check into version control.
FLUTTER_ROOT=$FLUTTER_ROOT
FLUTTER_APPLICATION_PATH=$APP_PATH
COCOAPODS_PARALLEL_CODE_SIGN=true
FLUTTER_TARGET=lib/main.dart
FLUTTER_BUILD_DIR=build
FLUTTER_BUILD_NAME=$BUILD_NAME
FLUTTER_BUILD_NUMBER=$BUILD_NUMBER
DART_OBFUSCATION=false
TRACK_WIDGET_CREATION=false
TREE_SHAKE_ICONS=true
PACKAGE_CONFIG=.dart_tool/package_config.json
XCCONFIG
  echo "[ci] Generated.xcconfig written (version $BUILD_NAME+$BUILD_NUMBER)"
fi

# ── CocoaPods ────────────────────────────────────────────────────────
echo "[ci] pod install..."
cd "$APP_PATH/ios"
pod install

echo "[ci] Setup complete."

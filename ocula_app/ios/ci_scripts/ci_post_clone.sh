#!/bin/sh
set -euo pipefail

REPO_PATH="${CI_PRIMARY_REPOSITORY_PATH:-$PWD}"
APP_PATH="$REPO_PATH/ocula_app"

if [ ! -d "$APP_PATH" ]; then
  APP_PATH="$REPO_PATH"
fi

if ! command -v flutter >/dev/null 2>&1; then
  echo "Flutter not found on PATH. Installing stable Flutter SDK..."
  git clone https://github.com/flutter/flutter.git --depth 1 -b stable "$HOME/flutter"
  export PATH="$HOME/flutter/bin:$PATH"
fi

echo "Using Flutter: $(command -v flutter)"
flutter --version

# Download iOS engine artifacts before building
flutter precache --ios

cd "$APP_PATH"

# Run a full Flutter iOS build so Generated.xcconfig and the CocoaPods
# xcfilelist files exist before Xcode Cloud's archive step loads the project.
# flutter pub get alone does not reliably produce Generated.xcconfig on CI.
flutter build ios --release --no-codesign

echo "Xcode Cloud post-clone setup completed."

import 'dart:io';
import 'dart:ui';
import 'package:tray_manager/tray_manager.dart';
import 'package:window_manager/window_manager.dart';

/// Manages the macOS menu bar / Windows system tray icon.
/// Only active on desktop platforms (macOS, Windows).
class TrayService with TrayListener, WindowListener {
  TrayService._();
  static final instance = TrayService._();

  bool _initialized = false;

  Future<void> init() async {
    if (_initialized) return;
    if (!_isDesktop) return;
    _initialized = true;

    // Window manager — prevent close from quitting; hide to tray instead.
    await windowManager.ensureInitialized();
    const options = WindowOptions(
      title: 'Ocula',
      center: true,
      minimumSize: Size(380, 600),
      skipTaskbar: false,
      titleBarStyle: TitleBarStyle.normal,
    );
    await windowManager.waitUntilReadyToShow(options);
    windowManager.addListener(this);
    await windowManager.setPreventClose(true);
    await windowManager.show();

    // Tray icon
    await trayManager.setIcon(_iconPath);
    await trayManager.setToolTip('Ocula');
    await _rebuildMenu();
    trayManager.addListener(this);
  }

  // ── TrayListener ────────────────────────────────────────────────────────────

  @override
  void onTrayIconMouseDown() {
    if (Platform.isWindows) _toggleWindow();
  }

  @override
  void onTrayIconRightMouseDown() {
    trayManager.popUpContextMenu();
  }

  @override
  void onTrayMenuItemClick(MenuItem item) {
    switch (item.key) {
      case 'show':
        _showWindow();
      case 'hide':
        _hideWindow();
      case 'quit':
        _quit();
    }
  }

  // ── WindowListener ───────────────────────────────────────────────────────────

  @override
  void onWindowClose() {
    // Hide to tray instead of quitting.
    _hideWindow();
  }

  @override
  void onWindowShow() {
    _rebuildMenu();
  }

  @override
  void onWindowHide() {
    _rebuildMenu();
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  Future<void> _toggleWindow() async {
    if (await windowManager.isVisible()) {
      _hideWindow();
    } else {
      _showWindow();
    }
  }

  Future<void> _showWindow() async {
    await windowManager.show();
    await windowManager.focus();
    await _rebuildMenu();
  }

  Future<void> _hideWindow() async {
    await windowManager.hide();
    await _rebuildMenu();
  }

  Future<void> _quit() async {
    await windowManager.setPreventClose(false);
    await windowManager.destroy();
  }

  Future<void> _rebuildMenu() async {
    final visible = await windowManager.isVisible();
    await trayManager.setContextMenu(Menu(
      items: [
        MenuItem(
          key: visible ? 'hide' : 'show',
          label: visible ? 'Hide Ocula' : 'Show Ocula',
        ),
        MenuItem.separator(),
        MenuItem(key: 'quit', label: 'Quit Ocula'),
      ],
    ));
  }

  bool get _isDesktop => Platform.isMacOS || Platform.isWindows;

  String get _iconPath {
    if (Platform.isMacOS) return 'assets/images/tray_icon_mac.png';
    return 'assets/images/tray_icon_win.png';
  }

  void dispose() {
    if (!_isDesktop) return;
    trayManager.removeListener(this);
    windowManager.removeListener(this);
  }
}

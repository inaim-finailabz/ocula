#!/usr/bin/env ruby
# Adds the OculaShare extension target to the Xcode project.
# Run: ruby add_share_extension.rb

$LOAD_PATH.unshift '/opt/homebrew/Cellar/cocoapods/1.16.2_1/libexec/gems/xcodeproj-1.27.0/lib'
# xcodeproj depends on these gems (bundled with CocoaPods)
Dir.glob('/opt/homebrew/Cellar/cocoapods/1.16.2_1/libexec/gems/*/lib').each { |p| $LOAD_PATH.unshift p }

require 'xcodeproj'

project_path = File.join(__dir__, 'Runner.xcodeproj')
project = Xcodeproj::Project.open(project_path)

# Check if target already exists
if project.targets.any? { |t| t.name == 'OculaShare' }
  puts "OculaShare target already exists — skipping."
  exit 0
end

# Create the extension target
ext_target = project.new_target(
  :app_extension,
  'OculaShare',
  :ios,
  '15.0'
)

# Set bundle identifier
ext_target.build_configurations.each do |config|
  config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.finailabz.ai.ocula.share'
  config.build_settings['INFOPLIST_FILE'] = 'OculaShare/Info.plist'
  config.build_settings['SWIFT_VERSION'] = '5.0'
  config.build_settings['CODE_SIGN_ENTITLEMENTS'] = 'OculaShare/OculaShare.entitlements'
  config.build_settings['DEVELOPMENT_TEAM'] = 'X77UL7V38R'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['GENERATE_INFOPLIST_FILE'] = 'NO'
  config.build_settings['CURRENT_PROJECT_VERSION'] = '1'
  config.build_settings['MARKETING_VERSION'] = '1.0.0'
  config.build_settings['TARGETED_DEVICE_FAMILY'] = '1,2'
end

# Add source files to the extension target
share_group = project.main_group.new_group('OculaShare', 'OculaShare')

swift_ref = share_group.new_file('OculaShare/ShareViewController.swift')
ext_target.source_build_phase.add_file_reference(swift_ref)

plist_ref = share_group.new_file('OculaShare/Info.plist')
entitlements_ref = share_group.new_file('OculaShare/OculaShare.entitlements')

# Add entitlements to main Runner target too
runner_target = project.targets.find { |t| t.name == 'Runner' }
if runner_target
  runner_target.build_configurations.each do |config|
    config.build_settings['CODE_SIGN_ENTITLEMENTS'] = 'Runner/Runner.entitlements'
  end
end

# Add OculaShare as a dependency of Runner (embeds the extension)
runner_target.add_dependency(ext_target) if runner_target

# Add embed extension build phase
embed_phase = runner_target.new_copy_files_build_phase('Embed App Extensions')
embed_phase.dst_subfolder_spec = '13' # PlugIns folder
embed_phase.add_file_reference(ext_target.product_reference)

project.save

puts "OculaShare extension target added successfully!"
puts "Don't forget to set up App Groups in Xcode Signing & Capabilities."

#!/usr/bin/env ruby
# Fix the OculaShare extension target in the Xcode project.

$LOAD_PATH.unshift '/opt/homebrew/Cellar/cocoapods/1.16.2_1/libexec/gems/xcodeproj-1.27.0/lib'
Dir.glob('/opt/homebrew/Cellar/cocoapods/1.16.2_1/libexec/gems/*/lib').each { |p| $LOAD_PATH.unshift p }

require 'xcodeproj'

project_path = File.join(__dir__, 'Runner.xcodeproj')
project = Xcodeproj::Project.open(project_path)

# Remove existing OculaShare target if present
ext_target = project.targets.find { |t| t.name == 'OculaShare' }
if ext_target
  puts "Removing existing OculaShare target..."
  ext_target.remove_from_project
end

# Remove existing OculaShare group if present
share_group = project.main_group.children.find { |g| g.display_name == 'OculaShare' }
share_group&.remove_from_project

# Remove existing embed phase from Runner
runner_target = project.targets.find { |t| t.name == 'Runner' }
runner_target.build_phases.select { |p| p.display_name == 'Embed App Extensions' }.each do |p|
  p.remove_from_project
end

# Remove dependency on OculaShare from Runner
runner_target.dependencies.select { |d| d.target&.name == 'OculaShare' || d.target.nil? }.each do |d|
  d.remove_from_project
end

# Now create fresh extension target
ext_target = project.new_target(
  :app_extension,
  'OculaShare',
  :ios,
  '15.0'
)

# Product name must match
ext_target.build_configurations.each do |config|
  config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.finailabz.ai.ocula.share'
  config.build_settings['PRODUCT_NAME'] = 'OculaShare'
  config.build_settings['INFOPLIST_FILE'] = 'OculaShare/Info.plist'
  config.build_settings['SWIFT_VERSION'] = '5.0'
  config.build_settings['CODE_SIGN_ENTITLEMENTS'] = 'OculaShare/OculaShare.entitlements'
  config.build_settings['DEVELOPMENT_TEAM'] = 'X77UL7V38R'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['GENERATE_INFOPLIST_FILE'] = 'NO'
  config.build_settings['CURRENT_PROJECT_VERSION'] = '1'
  config.build_settings['MARKETING_VERSION'] = '1.0.0'
  config.build_settings['TARGETED_DEVICE_FAMILY'] = '1,2'
  config.build_settings['SKIP_INSTALL'] = 'YES'
end

# Add source files to the extension target
share_group = project.main_group.new_group('OculaShare', 'OculaShare')

swift_ref = share_group.new_file('ShareViewController.swift')
ext_target.source_build_phase.add_file_reference(swift_ref)

share_group.new_file('Info.plist')
share_group.new_file('OculaShare.entitlements')

# Set entitlements on main Runner target
runner_target.build_configurations.each do |config|
  config.build_settings['CODE_SIGN_ENTITLEMENTS'] = 'Runner/Runner.entitlements'
end

# Add dependency — Runner depends on OculaShare
runner_target.add_dependency(ext_target)

# Add embed extensions build phase — insert BEFORE CocoaPods script phases
# to avoid Xcode build cycle with [CP] Embed Pods Frameworks / Thin Binary.
embed_phase = runner_target.new_copy_files_build_phase('Embed App Extensions')
embed_phase.dst_subfolder_spec = '13'  # PlugIns
build_file = embed_phase.add_file_reference(ext_target.product_reference)
build_file.settings = { 'ATTRIBUTES' => ['RemoveHeadersOnCopy'] }

# Move embed phase to just after the link phase (before script phases)
phases = runner_target.build_phases
phases.delete(embed_phase)
link_index = phases.index { |p| p.is_a?(Xcodeproj::Project::Object::PBXFrameworksBuildPhase) }
insert_at = link_index ? link_index + 1 : 0
phases.insert(insert_at, embed_phase)

project.save

puts "OculaShare extension target fixed successfully!"

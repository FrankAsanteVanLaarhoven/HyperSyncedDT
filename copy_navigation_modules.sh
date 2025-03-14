#!/bin/bash

# Script to copy all missing navigation modules to the backup

SOURCE_DIR="/Users/frankvanlaarhoven/Desktop/HyperSyncedDT/hypersyncdt/frontend"
TARGET_DIR="/Users/frankvanlaarhoven/Desktop/HyperSyncedDT_Backup_20250313_212222/frontend"

# List of module files to copy
modules=(
  "rag_agent_creator.py"
  "wear_pattern_recognition.py"
  "tool_life_prediction.py"
  "virtual_testing.py"
  "process_simulation.py"
  "what_if_analysis.py"
  "experiment_tracking.py"
  "digital_twin_simulation.py"
  "process_monitoring.py"
  "quality_control.py"
  "energy_optimization.py"
  "settings.py"
)

# Copy each module
for module in "${modules[@]}"; do
  echo "Copying $module..."
  cp "$SOURCE_DIR/$module" "$TARGET_DIR/" 2>/dev/null || echo "Failed to copy $module (might not exist)"
done

echo "All navigation modules have been copied!" 
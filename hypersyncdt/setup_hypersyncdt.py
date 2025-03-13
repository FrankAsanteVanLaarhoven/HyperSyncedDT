cat > ~/Desktop/setup_hypersyncdt.py << 'EOL'
import os
import shutil
import subprocess
from datetime import datetime

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error: {str(e)}")
        return False

def setup_project():
    # Setup paths
    desktop_path = os.path.expanduser("~/Desktop")
    project_path = os.path.join(desktop_path, "HyperSyncedDT")
    backup_path = os.path.join(desktop_path, f"HyperSyncedDT_Backup_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # Create backup
    print("Creating backup...")
    if os.path.exists(project_path):
        shutil.copytree(project_path, backup_path)
        print(f"✅ Backup created at: {backup_path}")

    # Setup git repository
    print("\nSetting up Git repository...")
    os.chdir(project_path)
    
    commands = [
        "git init",
        "git add .",
        'git commit -m "Initial commit: HyperSyncedDT - Advanced Digital Twin Platform"',
        "git remote add origin https://github.com/FrankAsanteVanLaarhoven/HyperSyncedDT.git",
        "git branch -M main",
        "git push -u origin main"
    ]

    for command in commands:
        print(f"\nExecuting: {command}")
        if not run_command(command):
            print("❌ Setup failed")
            return False

    print("\n✅ Project setup completed successfully!")
    print(f"Project location: {project_path}")
    print(f"Backup location: {backup_path}")
    return True

if __name__ == "__main__":
    setup_project()
EOL
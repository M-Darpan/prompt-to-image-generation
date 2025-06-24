import subprocess
import sys
import venv
from pathlib import Path

def setup_project():
    """Set up the project with a virtual environment and install dependencies"""
    # Create virtual environment
    venv_path = Path("venv")
    venv.create(venv_path, with_pip=True)
    
    # Get the path to the Python executable in the virtual environment
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        python_path = venv_path / "bin" / "python"
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    
    # Install dependencies
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    print("Setup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")

if __name__ == "__main__":
    setup_project() 
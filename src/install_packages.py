import subprocess
import sys
import os

def install_packages():
    # Install kokoro and soundfile
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kokoro", "soundfile"])
    
    # Install espeak-ng based on OS
    import platform
    system = platform.system()
    if system == "Linux":
        subprocess.check_call(["sudo", "apt-get", "-qq", "-y", "install", "espeak-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif system == "Darwin":  # macOS
        subprocess.check_call(["brew", "install", "espeak"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Espeak installation skipped for this OS. OOD words may not be handled properly.")
    
    # Install other required packages
    required_packages = [
        "torch",
        "transformers",
        "faster-whisper",
        "Pillow",
        "soundfile"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required_packages)

install_packages()

import subprocess

# Run Detect_platform-ratio.py
subprocess.run(["python", "Detect_platform-ratio.py"], check=True)

# Then run openpose_choosefile-show meter.py
subprocess.run(["python", "openpose_choosefile-show meter.py"], check=True)
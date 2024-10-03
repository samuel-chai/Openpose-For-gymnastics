import subprocess

# Run Detect02_platform-ratio.py
subprocess.run(["python", "Detect02_platform-ratio.py"], check=True)

# Then run Detect03-openpose_choosefile-show meter.py
subprocess.run(["python", "Detect03-openpose_choosefile-show meter.py"], check=True)
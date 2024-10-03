import subprocess

# Run Detect01-openpose_choosefile_structure.py
subprocess.run(["python", "Detect01-openpose_choosefile_structure.py"], check=True)

# Then run Detect02_platform-ratio.py
subprocess.run(["python", "Detect02_platform-ratio.py"], check=True)

# Finally, run Detect03-openpose_choosefile-show meter.py
subprocess.run(["python", "Detect03-openpose_choosefile-show meter.py"], check=True)
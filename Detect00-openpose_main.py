import subprocess

# Full path to the Python 3.7 interpreter
python37_path = "C:/Users/ccs95/AppData/Local/Programs/Python/Python37/python.exe"

# Full path to the Python 3.8 interpreter
python38_path = "C:/Users/ccs95/AppData/Local/Programs/Python/Python38/python.exe"

# Run Detect01-openpose_choosefile_structure.py with Python 3.7
# subprocess.run([python37_path, "Detect01-openpose_choosefile_structure.py"], check=True)

# Then run Detect02_platform-ratio.py with Python 3.8
subprocess.run([python38_path, "Detect02_platform-ratio.py"], check=True)

# Finally, run Detect03-openpose_choosefile-show meter.py with Python 3.7
subprocess.run([python37_path, "Detect03-openpose_choosefile-show meter.py"], check=True)
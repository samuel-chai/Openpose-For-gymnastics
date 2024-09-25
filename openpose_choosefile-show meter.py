import sys
import cv2
import os
from sys import platform
import argparse
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename

def process_video(video_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
        params["video"] = video_path
        params["num_gpu_start"] = 0

        # Starting OpenPose
        opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
        opWrapper.configure(params)
        opWrapper.execute()

    except Exception as e:
        print(e)
        sys.exit(-1)

# Select video file using a file dialog
Tk().withdraw()  # Hide the Tkinter main window
video_path = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])  # Show the file dialog and get the selected file path

if video_path:
    # Check if the selected file is AVI or MP4
    ext = os.path.splitext(video_path)[1]
    if ext.lower() == ".avi" or ext.lower() == ".mp4":
        # Ask the user for the platform pixels
        root = Tk()
        root.withdraw()  # Hide the Tkinter main window
        user_input = simpledialog.askstring("Input", "Enter a number:", parent=root)
        print(f"User entered: {user_input}")
        process_video(video_path)
    else:
        print("Invalid file format. Please select an AVI or MP4 file.")
else:
    print("No video file selected.")
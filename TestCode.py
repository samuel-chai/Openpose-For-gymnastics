import sys
import cv2
import os
from sys import platform
import argparse
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

# Add this code before the while loop
nose_points = []

def process_video(video_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)！
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
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
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

        # Starting OpenPose
        opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
        opWrapper.configure(params)
        opWrapper.start()

        # Process Video
        datum = op.Datum()
        video = cv2.VideoCapture(video_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            if datum.poseKeypoints.any():
                nose_point = datum.poseKeypoints[0][0][:2]
                nose_points.append(nose_point)

            for point in nose_points:
                cv2.circle(datum.cvOutputData, tuple(map(int, point)), 5, (0, 0, 255), -1)

            cv2.imshow("OpenPose", datum.cvOutputData)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

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
        process_video(video_path)
    else:
        print("Invalid file format. Please select an AVI or MP4 file.")
else:
    print("No video file selected.")
import sys
import cv2
import os
from sys import platform
import argparse
import tkinter as tk
import json
from tkinter import filedialog

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin'
            import pyopenpose as op
        else:
            sys.path.append('../../python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # Initialize Tkinter for file dialogs
    root = tk.Tk()
    root.withdraw()

    # Show an "Open" dialog box and return the path to the selected file
    image_path = filedialog.askopenfilename(
        title='Select Image',
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        print("No image selected.")
        sys.exit(0)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {"model_folder": "../../../models/"}

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

    # Save keypoints to JSON
    if datum.poseKeypoints is not None:
        # Convert to list
        keypoints = datum.poseKeypoints.tolist()
        keypoints_json = json.dumps(keypoints, indent=4)
    else:
        keypoints_json = json.dumps([])

    # Ask user for file save location
    save_path = filedialog.asksaveasfilename(
        title='Save Keypoints',
        filetypes=[("JSON files", "*.json")],
        defaultextension=".json"
    )

    if save_path:
        with open(save_path, 'w') as file:
            file.write(keypoints_json)
        print(f"Keypoints saved to {save_path}")
    else:
        print("Save operation cancelled.")

except Exception as e:
    print(e)
    sys.exit(-1)

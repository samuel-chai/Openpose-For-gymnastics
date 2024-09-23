import sys
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if sys.platform == "win32":
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

    # Extract keypoints
    keypoints = datum.poseKeypoints[0]

    # Convert keypoints to DataFrame
    df = pd.DataFrame(keypoints.reshape(-1, 3), columns=['x', 'y', 'c'])

    # Add keypoints names as a new column

    keypoint_names = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                      'left_wrist', 'hip',  'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                      'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel',
                      'right_big_toe', 'right_small_toe', 'right_heel']
    df['keypoint'] = keypoint_names[:len(df)]

    # Save DataFrame to Excel
    save_path = filedialog.asksaveasfilename(
        title='Save Excel',
        filetypes=[("Excel files", "*.xlsx")],
        defaultextension=".xlsx"
    )

    if save_path:
        df.to_excel(save_path, index=False)
        print(f"Keypoints saved to {save_path}")
    else:
        print("Save operation cancelled.")

    # Display Skeleton
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

except Exception as e:
    print(e)
    sys.exit(-1)

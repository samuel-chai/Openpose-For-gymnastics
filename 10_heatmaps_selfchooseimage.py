import sys
import cv2
import os
from sys import platform
import argparse
import tkinter as tk
from tkinter import filedialog

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Update these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin'
            import pyopenpose as op
        else:
            # Update these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing

    # Show an "Open" dialog box and return the path to the selected file
    image_path = filedialog.askopenfilename(
        title='Select Image',
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        print("No image selected.")
        sys.exit(0)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {
        "model_folder": "../../../models/",
        "heatmaps_add_parts": True,
        "heatmaps_add_bkg": True,
        "heatmaps_add_PAFs": True,
        "heatmaps_scale": 2
    }

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Process outputs
    outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
    outputImageF = cv2.merge([outputImageF[0, :, :], outputImageF[1, :, :], outputImageF[2, :, :]])
    outputImageF = (outputImageF * 255.).astype(dtype='uint8')
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    # Display Image
    counter = 0
    while True:
        num_maps = heatmaps.shape[0]
        heatmap = heatmaps[counter, :, :].copy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", combined)
        key = cv2.waitKey(-1)
        if key == 27:  # exit on ESC
            break
        counter = (counter + 1) % num_maps

except Exception as e:
    print(e)
    sys.exit(-1)

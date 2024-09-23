import sys
import cv2
import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image, ImageTk

def process_video(video_path, excel_save_path):
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

        # Starting OpenPose
        params = {"model_folder": "../../../models/", "render_pose": 1}
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Open video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create DataFrame to store data
        df = pd.DataFrame()

        # Process video frames
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            if datum.poseKeypoints is not None:  # Check if keypoints were detected
                keypoints = datum.poseKeypoints[0][:, :2]  # Extract x, y coordinates
            else:
                print(f"No keypoints detected in frame {frame_num}")
                keypoints = np.zeros((25, 2))  # Set keypoints to zeros if not detected

            # Extract relevant keypoints
            relevant_keypoints = keypoints[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

            # Flatten and append to DataFrame
            row_data = [frame_num / fps]
            row_data.extend(relevant_keypoints.flatten())
            df = df.append(pd.Series(row_data), ignore_index=True)

            # Display video frame
            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

        # Close video window
        cv2.destroyAllWindows()

        # Rename columns
        df.columns = ['Frame', 'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y',
                      'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y',
                      'Hip_x', 'Hip_y', 'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y',
                      'LHip_x', 'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y']

        # Reorder columns according to the provided sequence
        df = df[['Frame', 'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y',
                 'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y',
                 'Hip_x', 'Hip_y', 'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y',
                 'LHip_x', 'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y']]

        # Save DataFrame to Excel
        df.to_excel(excel_save_path, index=False)
        print(f"Keypoints saved to {excel_save_path}")

    except Exception as e:
        print(e)

# Process video and store keypoints data in the background
Tk().withdraw()  # Hide the Tkinter main window
video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])  # Show the file dialog and get the selected video file path

if video_path:
    # Ask user for file save location
    excel_save_path = filedialog.asksaveasfilename(
        title='Save Excel File',
        filetypes=[("Excel files", "*.xlsx")],
        defaultextension=".xlsx"
    )

    if excel_save_path:
        process_video(video_path, excel_save_path)
    else:
        print("Save operation cancelled.")
else:
    print("No video file selected.")

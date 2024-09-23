import sys
import cv2
import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog

def process_video(video_path_1, video_path_2, excel_save_path):
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

        # Create DataFrame to store data
        df = pd.DataFrame()

        for video_index, video_path in enumerate([video_path_1, video_path_2]):
            # Open video file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Process video frames
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Show video frame
                cv2.imshow('Video Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                if datum.poseKeypoints is not None:  # Check if keypoints were detected
                    keypoints = datum.poseKeypoints[0][:, :2]  # Extract x, y coordinates
                else:
                    print(f"No keypoints detected in frame {frame_num}")
                    keypoints = np.zeros((25, 2))  # Set keypoints to zeros if not detected

                # Extract left and right ankle keypoints
                left_ankle = keypoints[14]
                right_ankle = keypoints[11]

                # Append to DataFrame
                if video_index == 0:  # For the first video, save to columns A-D
                    df = df.append({'Frame': frame_num / fps, 'Left_Ankle_x': left_ankle[0], 'Left_Ankle_y': left_ankle[1],
                                    'Right_Ankle_x': right_ankle[0], 'Right_Ankle_y': right_ankle[1]}, ignore_index=True)
                else:  # For the second video, save to columns F-I
                    if 'Frame_2' not in df:
                        df['Frame_2'] = np.nan
                    if 'Left_Ankle_x_2' not in df:
                        df['Left_Ankle_x_2'] = np.nan
                    if 'Left_Ankle_y_2' not in df:
                        df['Left_Ankle_y_2'] = np.nan
                    if 'Right_Ankle_x_2' not in df:
                        df['Right_Ankle_x_2'] = np.nan
                    if 'Right_Ankle_y_2' not in df:
                        df['Right_Ankle_y_2'] = np.nan

                    df.at[frame_num, 'Frame_2'] = frame_num / fps
                    df.at[frame_num, 'Left_Ankle_x_2'] = left_ankle[0]
                    df.at[frame_num, 'Left_Ankle_y_2'] = left_ankle[1]
                    df.at[frame_num, 'Right_Ankle_x_2'] = right_ankle[0]
                    df.at[frame_num, 'Right_Ankle_y_2'] = right_ankle[1]

                frame_num += 1

            cap.release()  # Release the video capture object

        # Calculate differences and set negatives to 0
        df['d_left'] = np.maximum(df['Left_Ankle_x'] - df['Left_Ankle_x_2'], 0)
        df['d_right'] = np.maximum(df['Right_Ankle_x'] - df['Right_Ankle_x_2'], 0)

        # Add ankle Z columns
        df['Left_Ankle_Z'] = (895 * 434) / df['d_left']
        df['Right_Ankle_Z'] = (895 * 434) / df['d_right']

        # Rename columns
        df.rename(columns={'d_left': 'd_left', 'd_right': 'd_right',
                           'Left_Ankle_x': 'left ankle X', 'Left_Ankle_y': 'left ankle Y', 'Left_Ankle_Z': 'left ankle Z',
                           'Right_Ankle_x': 'right ankle X', 'Right_Ankle_y': 'right ankle Y', 'Right_Ankle_Z': 'right ankle Z',
                           'Left_Ankle_x_2': 'left ankle X_2', 'Left_Ankle_y_2': 'left ankle Y_2',
                           'Right_Ankle_x_2': 'right ankle X_2', 'Right_Ankle_y_2': 'right ankle Y_2'}, inplace=True)

        # Save DataFrame to Excel
        df.to_excel(excel_save_path, index=False)
        print(f"Data saved to {excel_save_path}")

    except Exception as e:
        print(e)

# Process two videos and store left and right ankle positions in the Excel file
Tk().withdraw()  # Hide the Tkinter main window
video_path_1 = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])  # Show the file dialog and get the selected video file path
video_path_2 = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])  # Show the file dialog and get the selected video file path

if video_path_1 and video_path_2:
    # Ask user for file save location
    excel_save_path = filedialog.asksaveasfilename(
        title='Save Excel File',
        filetypes=[("Excel files", "*.xlsx")],
        defaultextension=".xlsx"
    )

    if excel_save_path:
        process_video(video_path_1, video_path_2, excel_save_path)
    else:
        print("Save operation cancelled.")
else:
    print("No video files selected.")

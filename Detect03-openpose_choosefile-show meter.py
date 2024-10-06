# 這段code原先是要同時用yolo偵測跳馬台長度, 換算跳馬台的比例與Openpose人體移動的軌跡做結合
# 但是因為yolo的部分有點問題, 所以只用Openpose的部分
# 跳馬台預設的長度都是1.25m, 使用前要先輸入跳馬台藉由yolo偵測得到的長度, 後面才能跑數據

import sys
import cv2
import os
from sys import platform
import argparse
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename
import numpy as np
import pickle
import pandas as pd
from tkinter.filedialog import asksaveasfilename
from tkinter import filedialog


def adjust_coordinate(coord):
    return coord * pixel_ratio
def process_video(video_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                sys.path.append('../../python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Custom Params
        params = dict()
        params["model_folder"] = "../../../models/"
        params["video"] = video_path
        params["num_gpu_start"] = 0

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Create an empty DataFrame to store the data
        data = pd.DataFrame(columns=["Frame", "Time", "Hip_X", "Hip_Y"])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Get the nose keypoint (index 0)
            # 在主循環中修改坐標顯示部分
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                nose = datum.poseKeypoints[0][0]
                nose_x, nose_y = adjust_coordinate(nose[0]), adjust_coordinate(nose[1])

                left_hip = datum.poseKeypoints[0][8]
                right_hip = datum.poseKeypoints[0][11]

                hip_x = adjust_coordinate((left_hip[0] + right_hip[0]) / 2)
                hip_y = adjust_coordinate((left_hip[1] + right_hip[1]) / 2)

                left_ankle = datum.poseKeypoints[0][10]
                right_ankle = datum.poseKeypoints[0][13]

                left_elbow = datum.poseKeypoints[0][6]
                left_wrist = datum.poseKeypoints[0][7]

                # Calculate the angle between the line and the horizontal line
                dx = left_wrist[0] - left_elbow[0]
                dy = left_wrist[1] - left_elbow[1]

                if dx > 0 and dy >= 0:
                    angle = np.arctan(dy / dx)
                elif dx < 0 and dy >= 0:
                    angle = np.pi - np.arctan(dy / -dx)
                elif dx < 0 and dy < 0:
                    angle = np.pi + np.arctan(dy / dx)
                elif dx > 0 and dy < 0:
                    angle = 2 * np.pi - np.arctan(-dy / dx)
                else:
                    angle = 0

                angle = np.degrees(angle)  # Convert to degrees

                # 在畫面上顯示調整後的坐標
                cv2.putText(frame, f"Nose: ({nose_x:.2f}, {nose_y:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Hip: ({hip_x:.2f}, {hip_y:.2f})", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Left Ankle: ({adjust_coordinate(left_ankle[0]):.2f}, {adjust_coordinate(left_ankle[1]):.2f})",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Right Ankle: ({adjust_coordinate(right_ankle[0]):.2f}, {adjust_coordinate(right_ankle[1]):.2f})",
                            (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Arm Angle: {angle:.2f}", (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("OpenPose", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add the data of this frame to the DataFrame
            time = frame_count / cap.get(cv2.CAP_PROP_FPS)  # Calculate the time of this frame
            data = data.append({"Frame": frame_count, "Time": time, "Hip_X": hip_x, "Hip_Y": hip_y}, ignore_index=True)

            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()

        # Write the DataFrame to an Excel file
        root = Tk()
        root.withdraw()  # Hide the main window
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])  # Show the file dialog and get the selected file path
        if output_path:
            data.to_excel(output_path, index=False)
        else:
            print("Save operation cancelled.")

    except Exception as e:
        print(e)
        sys.exit(-1)

# Load the video_path value from the file
with open('output_video_path.pkl', 'rb') as f:
    video_path = pickle.load(f)

if video_path:
    ext = os.path.splitext(video_path)[1]
    # Load the average_width value from the file
    with open('average_width.pkl', 'rb') as f:
        platform_ratio = pickle.load(f)
    if ext.lower() in [".avi", ".mp4"]:
        root = Tk()
        root.withdraw()
        pixel_ratio = 1.25 / platform_ratio
        process_video(video_path)
    else:
        print("Invalid file format. Please select an AVI or MP4 file.")
else:
    print("No video file selected.")
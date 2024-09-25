import sys
import cv2
import os
from sys import platform
import argparse
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename
import numpy as np


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
                angle = np.arctan2(dy, dx)  # This returns radians
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

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        sys.exit(-1)

# Select video file using a file dialog
Tk().withdraw()
video_path = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

if video_path:
    ext = os.path.splitext(video_path)[1]
    if ext.lower() in [".avi", ".mp4"]:
        root = Tk()
        root.withdraw()
        platform_ratio = float(simpledialog.askstring("Input", "請輸入Platform像素比例:", parent=root))
        pixel_ratio = 1.25 / platform_ratio
        process_video(video_path)
    else:
        print("Invalid file format. Please select an AVI or MP4 file.")
else:
    print("No video file selected.")
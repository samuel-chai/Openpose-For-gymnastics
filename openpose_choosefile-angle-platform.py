import sys
import cv2
import os
from sys import platform
import argparse
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import torch
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes, print_args
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) and new unpadded shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def run_yolo(video_path, output_path, weights, data, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000,
             device='', augment=False, agnostic_nms=False, half=False, dnn=False):
    try:
        # Load YOLOv5 model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half precision
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        # Load video
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            success, img0 = cap.read()
            if not success:
                break

            # Resize and pad image while maintaining aspect ratio
            img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0

            pred = model(img, augment=augment, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms,
                                       max_det=max_det)

            # Process YOLOv5 predictions
            annotator = Annotator(img0, line_width=1, example=str(names))
            for i, det in enumerate(pred):  # per image
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        # 缩小框的范围，例如缩小 10%
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        x1 -= width * 0.05
                        y1 += height * 0.07
                        x2 -= width * 0.04
                        xyxy = [x1, y1, x2, y2]

                        # 绘制边界框和标签
                        label = f"{names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # Write the frame with detections to the output video
            out.write(img0)

            # Display the resulting frame
            cv2.imshow('YOLOv5 Detection', img0)
            if cv2.waitKey(1) == ord('q'):
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        sys.exit(-1)


def run_openpose(video_path, output_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ[
                                         'PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
        params["video"] = video_path
        params["num_gpu_start"] = 0
        params["render_pose"] = 1  # Enable rendering

        # Starting OpenPose
        opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
        opWrapper.configure(params)
        opWrapper.start()

        # Process video
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            success, img = cap.read()
            if not success:
                break

            datum = op.Datum()
            datum.cvInputData = img
            opWrapper.emplaceAndPop([datum])

            # Write the frame with detections to the output video
            out.write(datum.cvOutputData)

            # Display the resulting frame
            cv2.imshow('OpenPose Detection', datum.cvOutputData)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        sys.exit(-1)


# Select video file using a file dialog
Tk().withdraw()  # Hide the Tkinter main window
video_path = askopenfilename(
    filetypes=[("Video Files", "*.mp4;*.avi")])  # Show the file dialog and get the selected file path

if video_path:
    # Check if the selected file is AVI or MP4
    ext = os.path.splitext(video_path)[1]
    if ext.lower() == ".avi" or ext.lower() == ".mp4":
        # Select output video file for YOLOv5
        yolo_output_path = asksaveasfilename(defaultextension=".mp4",
                                             filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")],
                                             title="Save YOLOv5 Output")
        if yolo_output_path:
            # Run YOLOv5
            run_yolo(video_path, yolo_output_path, weights='best.pt', data='data/jumping.yaml')

            # Select output video file for OpenPose
            openpose_output_path = asksaveasfilename(defaultextension=".mp4",
                                                     filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")],
                                                     title="Save OpenPose Output")
            if openpose_output_path:
                # Run OpenPose on YOLOv5 output
                run_openpose(yolo_output_path, openpose_output_path)
            else:
                print("No output file selected for OpenPose.")
        else:
            print("No output file selected for YOLOv5.")
    else:
        print("Invalid file format. Please select an AVI or MP4 file.")
else:
    print("No video file selected.")
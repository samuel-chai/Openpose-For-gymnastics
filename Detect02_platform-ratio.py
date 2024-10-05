# 這段代碼主要是要偵測圖片中的物件，並且計算跳馬的寬比，並且將結果顯示在圖片上
# 這段代碼是基於YOLOv5的模型，並且使用PyTorch作為後端

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes, print_args
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import pickle



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

def select_video_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def save_video_file():
    root = Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.asksaveasfilename(
        title="Save video file",
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("MKV files", "*.mkv")]
    )
    return file_path

def run(weights, source, data, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000, device='', augment=False, agnostic_nms=False, half=False, dnn=False):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half precision
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Load video
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Failed to open video: {source}"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load the output_path value from the file
    with open('output_video_path.pkl', 'rb') as f:
        output_path = pickle.load(f)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_width = 0
    count = 0

    while True:
        success, img0 = cap.read()
        if not success:
            break
        # Get the current frame number and total frames
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Get the video's FPS (frames per second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the current time in seconds
        current_time = current_frame / fps

        # Display the current frame number and time in the bottom-left corner
        cv2.putText(img0, f"Frame: {int(current_frame)}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img0, f"Time: {current_time:.2f} sec", (10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize and pad image while maintaining aspect ratio
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0

        pred = model(img, augment=augment, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms, max_det=max_det)

        # Process YOLOv5 predictions
        annotator = Annotator(img0, line_width=1, example=str(names))
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # Calculate width of the bounding box
                    x1, y1, x2, y2 = map(int, xyxy)
                    width = x2 - x1
                    
                    # Accumulate width and count
                    total_width += width
                    count += 1
                    
                    # Display width in the top-left corner
                    text = f"Platform(pixels): {width}"
                    cv2.putText(img0, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Draw bounding box and label
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Write the frame with detections to the output video
        out.write(img0)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Detection', img0)
        if cv2.waitKey(1) == ord('q'):
            break

    # Calculate and print average width
    if count > 0:
        average_width = total_width / count
        print(f"Average width of bounding boxes: {average_width:.2f} pixels")
        # Save the average_width value to a file
        with open('average_width.pkl', 'wb') as f:
            pickle.dump(average_width, f)
    else:
        print("No bounding boxes detected.")

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default='best.pt', help="model path")
    parser.add_argument("--source", type=str, default='data/videos', help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default='data/jumping.yaml', help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=Path("runs/detect"), help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # Select video file
    video_path = select_video_file()
    if not video_path:
        print("No video selected. Exiting...")
        return

    output_path = save_video_file()
    # Save the video_path value to a fileㄔㄇ
    with open('output_video_path.pkl', 'wb') as f:
        pickle.dump(output_path, f)

    # Run detection
    run(opt.weights, video_path, opt.data, opt.imgsz, opt.conf_thres, opt.iou_thres, opt.max_det, opt.device, opt.augment, opt.agnostic_nms, opt.half, opt.dnn)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
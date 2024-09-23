import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_oks_and_ap_for_each_point(annotated_kps, openpose_kps, keypoint_weights, keypoint_order, scale, threshold=0.5):
    results = {}
    for idx, kp_name in enumerate(keypoint_order):
        annotated_kp = annotated_kps[0][idx]  # 获取第一层列表中的第 idx 个关键点
        openpose_kp = openpose_kps[0][idx]    # 同上
        sigma = keypoint_weights[kp_name]

        if len(annotated_kp) >= 3 and len(openpose_kp) >= 3:
            xg, yg, vg = annotated_kp[:3]  # 假定格式为 [x, y, confidence]
            xp, yp, vp = openpose_kp[:3]  # 同上
            # ... (后续计算保持不变)


        d2 = ((xg - xp) ** 2 + (yg - yp) ** 2) / (scale ** 2)
        s2 = 2 * (sigma * scale) ** 2
        oks = np.exp(-d2 / s2) if vg > 0 and vp > 0 else 0
        ap = 1 if oks >= threshold else 0

        results[kp_name] = {"OKS": oks, "AP": ap}
    return results

def calculate_average_oks_ap(results):
    total_oks = np.mean([value["OKS"] for value in results.values()])
    total_ap = np.mean([value["AP"] for value in results.values()])
    return total_oks, total_ap

def main():
    # Initialize tkinter
    root = tk.Tk()
    root.withdraw()

    # Define keypoint weights
    keypoint_weights = {
        'nose': 0.026, 'neck': 0.098,  # 示例：为 'neck' 指定一个默认值
        'right shoulder': 0.079, 'right elbow': 0.072, 'right wrist': 0.062,
        'left shoulder': 0.079, 'left elbow': 0.072, 'left wrist': 0.062,
        'middle hip': 0.107, 'right hip': 0.107, 'right knee': 0.087,
        'right ankle': 0.089, 'left hip': 0.107, 'left knee': 0.087,
        'left ankle': 0.089, 'right eye': 0.025, 'left eye': 0.025,
        'right ear': 0.035, 'left ear': 0.035, 'left big toe': 0.1,
        'left small toe': 0.1, 'left heel': 0.1, 'right big toe': 0.1,
        'right small toe': 0.1, 'right heel': 0.1
    }

    # Define keypoint order
    keypoint_order = [
        'nose', 'neck', 'right shoulder', 'right elbow', 'right wrist', 'left shoulder',
        'left elbow', 'left wrist', 'middle hip', 'right hip', 'right knee', 'right ankle',
        'left hip', 'left knee', 'left ankle', 'right eye', 'left eye', 'right ear',
        'left ear', 'left big toe', 'left small toe', 'left heel', 'right big toe',
        'right small toe', 'right heel'
    ]

    # Get user input for bounding box width and height in a single dialog
    bbox_size = simpledialog.askstring("Input", "Enter bounding box width and height (pixels) separated by a space:")
    bbox_width, bbox_height = map(float, bbox_size.split())
    scale = np.sqrt(bbox_width * bbox_height)
    # # Get scale size from user
    # bbox_width = simpledialog.askfloat("Input", "Enter bounding box width (pixels):", parent=root)
    # bbox_height = simpledialog.askfloat("Input", "Enter bounding box height (pixels):", parent=root)
    # scale = np.sqrt(bbox_width * bbox_height)

    # Get annotated keypoints JSON file
    annotated_json_path = filedialog.askopenfilename(
        title='Select Annotated Keypoints JSON',
        filetypes=[("JSON files", "*.json")]
    )
    annotated_keypoints = read_json(annotated_json_path)

    # Get OpenPose keypoints JSON file
    openpose_json_path = filedialog.askopenfilename(
        title='Select OpenPose Keypoints JSON',
        filetypes=[("JSON files", "*.json")]
    )
    openpose_keypoints = read_json(openpose_json_path)

    # Calculate OKS and AP for each keypoint
    results = calculate_oks_and_ap_for_each_point(
        annotated_keypoints, openpose_keypoints, keypoint_weights, keypoint_order, scale
    )

    print(f"Scale (sqrt of width x height): {scale}")
    # Output results for each keypoint
    for kp_name, values in results.items():
        print(f"{kp_name}: OKS = {values['OKS']}, AP = {values['AP']}")

    # Calculate and output the average OKS and AP for all body points
    average_oks, average_ap = calculate_average_oks_ap(results)
    print(f"\nAverage OKS for all body points: {average_oks}")
    print(f"Average AP for all body points: {average_ap}")


# Run the main function
if __name__ == "__main__":
    main()

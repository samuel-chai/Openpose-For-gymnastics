import cv2
import json
import tkinter as tk
from tkinter import filedialog

# Initialize the list to store coordinates
points = []

# Extended keypoint order according to the provided sequence
keypoints = [
    "Nose", "Neck",
    "R-Sho", "R-Elb", "R-Wr", "L-Sho", "L-Elb", "L-Wr",
    "Mid-Hip", "R-Hip", "R-Knee", "R-Ank",
    "L-Hip", "L-Knee", "L-Ank",
    "R-Eye", "L-Eye", "R-Ear", "L-Ear",
    "L-BigToe", "L-SmallToe", "L-Heel",
    "R-BigToe", "R-SmallToe", "R-Heel"
]

# Index to keep track of the current keypoint
index = 0

# Function to be called when mouse is clicked
def draw_circle(event, x, y, flags, param):
    global index, points, img
    if event == cv2.EVENT_LBUTTONDOWN and index < len(keypoints):
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        points.append((x, y))
        print(f"Marked {keypoints[index]} at position {points[-1]}")
        index += 1

# Set up tkinter root
root = tk.Tk()
root.withdraw()

# Show an "Open" dialog box and return the path to the selected file
file_path = filedialog.askopenfilename()

# Ensure a file was selected
if file_path:
    img = cv2.imread(file_path)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)

    while True:
        # Display the current keypoint to be marked
        if index < len(keypoints):
            img_with_text = img.copy()
            text = f"Mark: {keypoints[index]}"
            cv2.putText(img_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('image', img_with_text)
        else:
            cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or index == len(keypoints):  # Quit if 'q' is pressed or all keypoints are marked
            break

    cv2.destroyAllWindows()

    # Save to a JSON file
    if points:
        # Append a default confidence score of 1
        keypoints_with_confidence = [list(point) + [1] for point in points]
        # Wrap keypoints data in an outer list, mimicking OpenPose output for multiple people
        all_people_keypoints = [keypoints_with_confidence]

        json_data = json.dumps(all_people_keypoints, indent=4)
        save_path = filedialog.asksaveasfilename(defaultextension=".json")
        if save_path:
            with open(save_path, 'w') as json_file:
                json_file.write(json_data)
            print(f"Keypoints saved to {save_path}")
        else:
            print("Save file operation cancelled.")
    else:
        print("No keypoints were marked.")
else:
    print("No file was selected.")

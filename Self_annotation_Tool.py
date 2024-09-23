import cv2
import tkinter as tk
from tkinter import simpledialog, filedialog

# Initialize global variables
annotation_point = None
frame_interval = None
annotated_points = []
point_marked = False  # Variable to track if a point has been marked


# Function to select the annotation point
def select_annotation_point():
    options = ["head", "right wrist", "left wrist", "right elbow", "left elbow",
               "back", "right knee", "left knee", "right ankle", "left ankle"]
    return simpledialog.askstring("Select Point", "Enter the point to annotate:\n" + "\n".join(options))

# Function to select the frame interval
def select_frame_interval():
    return simpledialog.askinteger("Frame Interval", "Enter the frame interval for annotation:", minvalue=1)

# Function to select the video file
def select_video_file():
    return filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4 *.avi *.mov")])

# Function to handle mouse clicks and annotate the video
def click_event(event, x, y, flags, param):
    global point_marked
    if event == cv2.EVENT_LBUTTONDOWN:
        annotated_points.append((x, y, param))
        point_marked = True
# Function to handle mouse clicks and annotate the video
# Function to annotate the video
# Function to annotate the video
def annotate_video(video_file, point_to_annotate, frame_interval):
    global point_marked
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    count = 0
    all_marked_points = []  # Store all marked points
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()  # Store the last frame for final output

        # Skip frames based on the interval
        if count % frame_interval != 0:
            count += 1
            continue

        # Reset point_marked
        point_marked = False

        # Draw all previous marked points on the frame
        for pt in all_marked_points:
            cv2.circle(frame, (pt[0], pt[1]), 2, (0, 0, 255), -1)  # Reduced radius to half

        # Show the frame with instructions
        cv2.putText(frame, f"Mark the {point_to_annotate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        cv2.setMouseCallback('Video', click_event, count)

        while not point_marked:
            key = cv2.waitKey(1) & 0xFF
            # Exit if 'q' is pressed
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            # Skip frame if space is pressed
            if key == ord(' '):
                break

        if point_marked:
            all_marked_points.append(annotated_points[-1][:2])  # Add the latest point

        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return last_frame, all_marked_points

# Function to save the annotated data
def save_annotation_data():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            for point in annotated_points:
                file.write(f"{point[0]}, {point[1]}, Frame: {point[2]}\n")
        print(f"Data saved to {file_path}")

def main():
    root = tk.Tk()
    root.withdraw()

    # Step 1: Select the point to annotate
    global annotation_point
    annotation_point = select_annotation_point()
    if not annotation_point:
        return

    # Step 2: Select the frame interval
    global frame_interval
    frame_interval = select_frame_interval()
    if not frame_interval:
        return

    # Step 3: Select the video file
    video_file = select_video_file()
    if not video_file:
        return

    # Step 4: Annotate the video and get the last frame
    last_frame, all_marked_points = annotate_video(video_file, annotation_point, frame_interval)

    # Step 5: Draw trajectory on the last frame and save it
    for i in range(1, len(all_marked_points)):
        cv2.line(last_frame, all_marked_points[i - 1], all_marked_points[i], (0, 255, 0), 2)
    output_image_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if output_image_path:
        cv2.imwrite(output_image_path, last_frame)
        print(f"Last frame with trajectory saved to {output_image_path}")

    # Step 6: Save the annotated data
    save_annotation_data()

if __name__ == "__main__":
    main()

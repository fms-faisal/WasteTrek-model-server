import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import torchvision
from torchvision.ops import nms

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the trained garbage detection model
def load_model(model_path, num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)  # Use weights instead of pretrained
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Set weights_only=True
    model.to(device)
    model.eval()
    return model

# Apply Non-Maximum Suppression (NMS)
def apply_nms(predictions, iou_threshold=0.5):
    for prediction in predictions:
        keep = nms(prediction['boxes'], prediction['scores'], iou_threshold)
        prediction['boxes'] = prediction['boxes'][keep]
        prediction['scores'] = prediction['scores'][keep]
        prediction['labels'] = prediction['labels'][keep]
    return predictions

# Visualize the results and draw bounding boxes
def draw_boxes_on_frame(frame, predictions, threshold=0.5):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
        if score > threshold:
            box = box.cpu().numpy().astype(int)
            draw.rectangle(box.tolist(), outline='red', width=2)

    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Process video frame by frame and save output with reduced FPS (5 FPS)
def process_video(input_video_path, output_video_path, model, target_fps=5):
    cap = cv2.VideoCapture(input_video_path)
    
    # Check if the video is opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        return
    
    # Read frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle case where FPS might be 0 or invalid
    if original_fps == 0:
        original_fps = 30  # Fallback FPS if reading from video fails
        print(f"Warning: FPS from video is 0, setting original FPS to {original_fps}")

    frame_interval = int(original_fps // target_fps)  # Interval to skip frames for 5 FPS
    print(f"Video Properties - Width: {width}, Height: {height}, Original FPS: {original_fps}, Target FPS: {target_fps}, Frame Interval: {frame_interval}")

    # Video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open output video writer {output_video_path}")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every nth frame (based on frame_interval)
        if frame_count % frame_interval == 0:
            # Convert frame to tensor
            frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

            # Get predictions
            with torch.no_grad():
                predictions = model(frame_tensor)
                predictions = apply_nms(predictions)

            # Draw boxes on the frame if garbage is detected
            if len(predictions[0]['boxes']) > 0:
                frame_with_boxes = draw_boxes_on_frame(frame, predictions)
            else:
                frame_with_boxes = frame

            # Write the frame with boxes into the output video
            out.write(frame_with_boxes)
        
        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

# Usage example
model_path = 'model_epoch_7.pth'  # Update this with the actual path to the model
input_video = 'testVideo.mp4'   # Input video path
output_video = 'output_with_garbage_boxes_5fps.mp4'  # Output video path

# Load the model and process the video
model = load_model(model_path)
process_video(input_video, output_video, model, target_fps=5)

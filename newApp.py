import os
import torch
from flask import Flask, request, jsonify
import PIL
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import torchvision
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the trained model
def load_trained_model(num_classes, model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Custom NMS function
def custom_nms(outputs, iou_threshold=0.5):
    processed_outputs = []
    for output in outputs:
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']

        # Apply NMS
        keep = nms(boxes, scores, iou_threshold)

        # Filter the results
        output = {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }
        processed_outputs.append(output)
    return processed_outputs

# Visualize predictions and draw bounding boxes
def visualize_predictions(model, image_tensor, frame_number, iou_threshold=0.5):
    with torch.no_grad():
        prediction = model(image_tensor)

    # Apply custom NMS
    prediction = custom_nms(prediction, iou_threshold=iou_threshold)

    # Convert the image tensor back to PIL Image for visualization
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_pil = Image.fromarray((img_np * 255).astype('uint8'))
    draw = ImageDraw.Draw(img_pil)

    if len(prediction[0]['boxes']) == 0:
        print(f"No garbage found in frame {frame_number}")
    else:
        print(f"Garbage found in frame {frame_number}")
        # Draw predicted boxes in red
        for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
            if score > 0.5:  # Threshold for drawing the box
                box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box
                if all(isinstance(coord, (int, float)) for coord in [x_min, y_min, x_max, y_max]):
                    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

    return img_pil

# Define the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model_path = 'model_epoch_7.pth'  # Change this to your model path
model = load_trained_model(model_path=model_path, num_classes=2)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Read the video
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_with_bbox.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a tensor
        image_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        # Visualize predictions and print garbage detection status
        img_pil = visualize_predictions(model, image_tensor, frame_number)

        # Convert the PIL image back to OpenCV format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(img_cv)

        frame_number += 1

    cap.release()
    out.release()

    return jsonify({'prediction': 'Garbage detection completed', 'output_video_path': output_path})

if __name__ == "__main__":
    import threading
    def run_app():
        app.run(debug=True, port=8080, use_reloader=False)
    threading.Thread(target=run_app).start()

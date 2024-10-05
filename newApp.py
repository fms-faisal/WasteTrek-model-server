import os
import torch
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import torchvision
from torchvision import transforms
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

import torch
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

# Visualize predictions and save images
def visualize_predictions(model, image_path, save_path, iou_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    # Apply custom NMS
    prediction = custom_nms(prediction, iou_threshold=iou_threshold)

    # Convert the image tensor back to PIL Image for visualization
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_pil = Image.fromarray((img_np * 255).astype('uint8'))
    draw = ImageDraw.Draw(img_pil)

    # Draw predicted boxes in red
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > 0.5:  # You can adjust this threshold if needed
            box = box.cpu().numpy()
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

    # Save the image with predictions
    img_pil.save(save_path)
    return image_tensor, prediction

# Define the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = 'model_epoch_7.pth'  # Change this to your model path
model = load_trained_model(model_path=model_path, num_classes=2)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_with_bbox.jpg')
    image_tensor, prediction = visualize_predictions(model, file_path, save_path)

    return jsonify({'prediction': 'Garbage' if len(prediction[0]['boxes']) > 0 else 'No Garbage', 'output_image_path': save_path})

if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)
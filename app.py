import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms

# SEBlock definition
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# ResidualBlock with SEBlock for enhanced features
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return torch.relu(out)

# Advanced CNN model definition
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=2):  # Set to 2 for binary classification
        super(AdvancedCNN, self).__init__()
        self.layer1 = self._make_layer(3, 64)  # Assume input channels = 3 (e.g., RGB)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)

        # Adjust the input size of the fully connected layer based on the flattened output
        self.fc = nn.Linear(512 * 45 * 45, num_classes)  # Adjust based on your model's architecture

    def _make_layer(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, downsample=downsample),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten the feature map
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model
num_classes = 2  # Set this to 2
model = AdvancedCNN(num_classes=num_classes)

# Load the model state dictionary
try:
    model.load_state_dict(torch.load(r'C:\Users\88018\Desktop\AndroidApp\Model-server\model_epoch_15.pth', map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except RuntimeError as e:
    print("Error loading the model:", e)

# Set the model to evaluation mode
model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((180, 180)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    try:
        # Open the image file
        image = Image.open(file.stream).convert('RGB')  # Ensure image is RGB
        image = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)  # Get the predicted class index
            predicted_class = predicted.item()

        return jsonify({'predicted_class': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

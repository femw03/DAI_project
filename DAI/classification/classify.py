import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2

class TrafficLightClassifier(nn.Module):
    def __init__(self):
        super(TrafficLightClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 4)  # 4 classes: back, green, red, yellow

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv2D -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))  # Dense layer
        x = self.fc2(x)  # Output layer with softmax activation
        return F.log_softmax(x, dim=1)

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    return img

def predict_class(img, model):
    img = preprocess_image(img)
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def classify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficLightClassifier().to(device)
    model.load_state_dict(torch.load('../classification/final_Carla_2.pth', map_location=device))
    model.eval()

    snip_dir = '../classification/Detected_Traffic_Lights'
    processed_images = set()

    while True:
        for img_file in os.listdir(snip_dir):
            img_path = os.path.join(snip_dir, img_file)
            if img_path not in processed_images:
                img = cv2.imread(img_path)
                if img is None:
                    print(f'Warning: Failed to load image {img_path}')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                class_id = predict_class(img, model)
                label_map = {0: 'back', 1: 'green', 2: 'red', 3: 'yellow'}
                label = label_map[class_id]
                print(f'Image: {img_file}, Predicted label: {label}')
                processed_images.add(img_path)
        time.sleep(0.01)  # Check for new images every 0.01 second

if __name__ == "__main__":
    classify()
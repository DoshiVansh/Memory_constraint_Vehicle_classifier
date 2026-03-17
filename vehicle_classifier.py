import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
# from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from PIL import Image

# -----------------------------
# Class Index Mapping (consistent across all submissions)
# -----------------------------
CLASS_IDX = {
    0: "Bus",
    1: "Truck",
    2: "Car",
    3: "Bike",
    4: "None"
}

# -----------------------------
# Lightweight Student Model (< 5 MB)
# -----------------------------
class StudentShuffleNet(nn.Module):
    def __init__(self, num_classes=5):
        super(StudentShuffleNet, self).__init__()
        
        # Load the highly efficient ShuffleNet V2
        self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        
        # Replacing the 1000-class head drops the size from 9.1 MB to 4.80 MB
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        
        # Ensure your StudentShuffleNet class is using shufflenet_v2_x1_0!
        self.model = StudentShuffleNet(num_classes=len(CLASS_IDX))
        
        if model_path:
            # 1. load the FP16 file into the 'state_dict' variable.
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # 2. Briefly convert our blank PyTorch model to FP16 to accept the FP16 weights
            self.model.half()
            
            # 3. Load the dictionary we defined in step 1
            self.model.load_state_dict(state_dict)
            
            # 4. Convert the populated model back to standard FP32 for perfectly safe CPU inference
            self.model.float()
        
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str) -> int:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    classifier = VehicleClassifier(model_path="student_model.pth")
    idx = classifier.predict("test_image.jpg")
    print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")
"""Model definitions and architecture factories."""

CLASS_IDX = {0: "Bus", 1: "Truck", 2: "Car", 3: "Bike", 4: "None"}

try:
    from torch import nn
    from torchvision.models import shufflenet_v2_x1_0
except ImportError:
    nn = None
    shufflenet_v2_x1_0 = None


class StudentShuffleNet(nn.Module if nn is not None else object):
    """ShuffleNet V2 x1.0 student with the project's five-class head."""

    def __init__(self, num_classes: int = len(CLASS_IDX)) -> None:
        super().__init__()
        self.model = shufflenet_v2_x1_0(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, images):
        return self.model(images)


def build_student_model(num_classes: int = len(CLASS_IDX)) -> StudentShuffleNet:
    """Create an uninitialized architecture without downloading weights."""
    return StudentShuffleNet(num_classes=num_classes)

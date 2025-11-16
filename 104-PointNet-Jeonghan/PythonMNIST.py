import torch, time, json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=7*7*64, out_features=10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def eval_mnist(src_image, weights_json, iteration, device, jitCompile):
    mnistNet = MnistNet().to(device).eval()

    if jitCompile:
        mnistNet = torch.compile(mnistNet)

    # JSON에서 가중치 읽어와 복사 (바로 device로)
    mnistNet.conv1.weight.data.copy_(
        torch.tensor(weights_json["layer1.0.weight"], dtype=torch.float32, device=device))
    mnistNet.conv1.bias.data.copy_(
        torch.tensor(weights_json["layer1.0.bias"], dtype=torch.float32, device=device))
    mnistNet.conv2.weight.data.copy_(
        torch.tensor(weights_json["layer2.0.weight"], dtype=torch.float32, device=device))
    mnistNet.conv2.bias.data.copy_(
        torch.tensor(weights_json["layer2.0.bias"], dtype=torch.float32, device=device))
    mnistNet.fc.weight.data.copy_(
        torch.tensor(weights_json["fc.weight"], dtype=torch.float32, device=device))
    mnistNet.fc.bias.data.copy_(
        torch.tensor(weights_json["fc.bias"], dtype=torch.float32, device=device))

    H, W, C = src_image.shape
    image_tensor = torch.tensor(src_image, dtype=torch.float32, device=device).reshape(1, C, H, W)

    with torch.no_grad():
        for _ in range(iteration):
            result = mnistNet(image_tensor)

    return result

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(dir + "/weights.json", "r") as f:
        weights_json = json.load(f)

    img = Image.open(dir + "/0.png").convert("L").resize((28, 28))
    src_image = np.array(img, dtype=np.float32).reshape((28, 28, 1)) / 255.0
    
    iteration = 10000
    device_type = "cuda"
    jitCompile = False

    available = (torch.cuda.is_available() if device_type == "cuda"
                 else torch.xpu.is_available() if device_type == "xpu"
                 else torch.is_vulkan_available() if device_type == "vulkan"
                 else False)
    device = torch.device(device_type if available else "cpu")

    torch.empty(1).to(device).cpu() # for device initialization

    start_time = time.time()
    result = eval_mnist(src_image, weights_json, iteration, device, jitCompile)
    end_time = time.time()

    print(f"[(PyTorch) MNIST evaluation: {iteration} iterations] => {(end_time - start_time)*1000:.3f}ms")
    print(result.cpu())

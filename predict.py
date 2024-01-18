import torch
from torch.nn.functional import softmax
import torchvision
import numpy
from PIL import Image
from argparse import ArgumentParser

symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

parser = ArgumentParser()
parser.add_argument("-w", "--weights", type=str, required=True)
parser.add_argument("input")
args = parser.parse_args()

with open(f"weights/{args.weights}.state", "rb") as file:
    weights = torch.load(file)

model = torchvision.models.mobilenet_v3_small(num_classes=26 + 10)
model = model.to("cuda")

model.load_state_dict(weights)

image = Image.open(args.input)
image = image.convert("RGB")
x = numpy.array(image, dtype=numpy.float32)
x = x.swapaxes(0, 2)
x = x[None]
x = torch.tensor(x, dtype=torch.float32, device="cuda")

model.eval()
with torch.no_grad():
    y = model(x)
    i = y[0].argmax()

print(symbols[i])
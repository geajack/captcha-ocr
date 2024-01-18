import torch
from torch.nn.functional import softmax
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from argparse import ArgumentParser
import pygame
import pygame.gfxdraw
import pygame.freetype

class Rectangle:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Dimensions:

    def __init__(self, w, h):
        self.w = w
        self.h = h


symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

parser = ArgumentParser()
parser.add_argument("-w", "--weights", type=str, required=True)
parser.add_argument("input")
args = parser.parse_args()

with open(f"weights/{args.weights}.state", "rb") as file:
    weights = torch.load(file)

model = torchvision.models.mobilenet_v3_small(num_classes=26 + 10 + 1)
model = model.to("cuda")
model.load_state_dict(weights)

image = Image.open(args.input)
image = image.convert("RGB")
x = pil_to_tensor(image)
x = x.to(dtype=torch.float32)

slices = x.unfold(2, 50, 1).to("cuda")
slices = slices.swapaxes(0, 2)
slices = slices.swapaxes(1, 2)
slices = slices.swapaxes(-1, -2)

model.eval()
with torch.no_grad():
    logits = model(slices)
    p = softmax(logits, dim=-1)
window_dimensions = Dimensions(1200, 900)
window = pygame.display.set_mode((window_dimensions.w, window_dimensions.h))

captcha_pixels = Rectangle(0, 0, image.width * 4, 200)
image = image.resize((captcha_pixels.w, captcha_pixels.h))
pygame_image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)

pygame.freetype.init()
font = pygame.freetype.Font("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", size=25)

axis_length = image.width
x_offset = 10
axis_label = pygame.Surface((axis_length + x_offset, 25))
axis_label.fill((0, 0, 0))
for i, symbol in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?"):
    text_surface, rectangle = font.render(symbol, (225, 225, 225))
    width = rectangle.width
    x = x_offset + i * (axis_length / 37) - width / 2
    axis_label.blit(text_surface, (x, 0))

graph = pygame.Surface((captcha_pixels.w, 200))
graph.fill((0, 0, 0, 0))
graph_x_offset = captcha_pixels.h // 2
pygame.draw.line(graph, (255, 255, 255), (graph_x_offset, 199), (captcha_pixels.w - graph_x_offset, 199))

points = []
p0 = None
p1 = None
current_index = None
for i in range(len(p)):
    value = float(p[i].max())
    index = int(p[i].argmax())
    p1 = (graph_x_offset + i * 4, int(200 - 200 * value))
    if p0 is not None:
        pygame.draw.line(
            graph,
            (255, 255, 255),
            p0, p1
        )
    p0 = p1

captcha_pixels.x = 600 - image.width / 2
captcha_pixels.y = 50
sliding_window_x = 0

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEMOTION:
            if not pygame.mouse.get_pressed()[0]:
                continue

            mx, my = pygame.mouse.get_pos()
            if not (0 <= mx - captcha_pixels.x <= image.width):
                continue
            if not (0 <= my - captcha_pixels.y <= image.height):
                continue

            sliding_window_x = int((mx - captcha_pixels.x) // 4) - 25
            if sliding_window_x < 0:
                sliding_window_x = 0
            
            if sliding_window_x > 200:
                sliding_window_x = 200
    
    window.fill((20, 20, 20))
    pygame.draw.rect(window, (150, 150, 150), (captcha_pixels.x - 10, captcha_pixels.y - 10, captcha_pixels.w + 20, captcha_pixels.h + 20), border_radius=15)
    
    pygame.draw.rect(window, (255, 255, 255), (captcha_pixels.x, captcha_pixels.y, captcha_pixels.w, captcha_pixels.h))
    window.blit(pygame_image, (captcha_pixels.x, captcha_pixels.y), special_flags=pygame.BLEND_RGB_SUB)

    x0 = captcha_pixels.x + 4*sliding_window_x
    sliding_box_height = captcha_pixels.h - 4
    pygame.draw.lines(
        window,
        color=(0, 255, 255),
        closed=True,
        points=[
            (x0, captcha_pixels.y),
            (x0 + sliding_box_height, captcha_pixels.y),
            (x0 + sliding_box_height, captcha_pixels.y + sliding_box_height),
            (x0, captcha_pixels.y + sliding_box_height)
        ],
        width=4
    )

    midline_x = captcha_pixels.x + 4*sliding_window_x + captcha_pixels.h // 2
    midline = 100
    pygame.draw.line(window, (225, 225, 225), (midline_x, captcha_pixels.y + captcha_pixels.h + 40), (midline_x, captcha_pixels.y + captcha_pixels.h + 40 + 199))
    window.blit(graph, (captcha_pixels.x, captcha_pixels.y + captcha_pixels.h + 40), special_flags=pygame.BLEND_RGBA_ADD)

    window.blit(axis_label, (captcha_pixels.x - x_offset, window_dimensions.h - 30))
    for i in range(37):
        pygame.draw.rect(
            window,
            (225, 225, 225),
            (
                captcha_pixels.x + i * (axis_length / 37) - ((axis_length / 37) / 2),
                window_dimensions.h - 30 - 10 - int(150 * p[sliding_window_x][i]),
                (axis_length / 37),
                int(150 * p[sliding_window_x][i])
            )
        )

    pygame.display.flip()
    clock.tick(60)
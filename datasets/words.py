from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, random
from pathlib import Path
from argparse import ArgumentParser
import torch

from datasets import pointillize, fonts, size

def random_image(text):
    image = Image.new("RGB", (5 * size, size), color=(255, 255, 255))
    artist = ImageDraw.Draw(image)

    font_index = randint(0, len(fonts) - 1)
    font = fonts[font_index]

    total_width = 0
    line_top = image.height
    line_bottom = 0
    for character in text:
        left, top, right, bottom = artist.textbbox((0, 0), character, font=font)
        width = right - left
        total_width += width
        line_top = min(line_top, top)
        line_bottom = max(line_bottom, bottom)

    line_height = line_bottom - line_top

    scaling = 50 / image.height
    x = (image.width - total_width) / 2
    y = random() * (image.height - line_height) - line_top
    boxes = []
    for character in text:
        artist.text((x, y), text=character, font=font, fill=(0, 0, 0))
        left, top, right, bottom = artist.textbbox((x, y), character, font=font)
        x = right
        boxes.append(
            (character, int(scaling * left), int(scaling * right))
        )
        
    image = pointillize(image)

    image = image.resize((int(image.width * scaling), 50))

    return image, boxes

if __name__ == "__main__":
    n_data = 3000
    image_tensor = torch.empty((n_data, 3, 250, 50))
    labels = []

    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for i in range(n_data):
        indices = [randint(0, len(symbols) - 1) for _ in range(5)]
        text = [symbols[i] for i in indices]
        text = str.join("", text)
        image, boxes = random_image(text)

        image.save(f"data/words/{i:05}.png")
        image_tensor[i] = torch.tensor(np.array(image).swapaxes(0, 2))
        labels.append(boxes)
        print(f"[{i + 1}/{n_data}]", end="        \r")

    print()

    with open("data/words/dataset.pickle", "wb") as out_file:
        torch.save((image_tensor, labels), out_file)
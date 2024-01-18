from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, random
from pathlib import Path
from argparse import ArgumentParser

from datasets import pointillize, fonts, size


def random_image(text, noise=1):
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    artist = ImageDraw.Draw(image)
    font = fonts[randint(0, len(fonts) - 1)]

    left, top, right, bottom = artist.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top

    x_gap = size / 2 - width / 2
    y_gap = size / 2 - height / 2
    dx, dy = 0.5 * x_gap * (2*random() - 1), 0.5 * y_gap * (2*random() - 1)
    artist.text((-left + x_gap + dx, -top + y_gap + dy), text, font=font, fill=(0, 0, 0))

    noisy_image = pointillize(image)    
    composite = noise * np.array(noisy_image) + (1 - noise) * np.array(image)
    image = Image.fromarray(composite.astype(np.uint8))
    
    image = image.resize((50, 50))

    return image


def generate(name):
    root = Path(f"data/{name}/")
    root.mkdir(parents=True, exist_ok=True)
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    csv = ""
    for i in range(10000):
        label = symbols[randint(0, len(symbols) - 1)]
        image = random_image(label, noise=1)
        filename = f"{i:05}.png"
        image.save(root / filename)
        csv += f"{filename},{label}\n"

    with open(root / "labels.csv", "w") as file:
        file.write(csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", type=str, nargs="?")
    args = parser.parse_args()

    if args.g:
        generate(args.g)
    else:
        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        label = symbols[randint(0, len(symbols) - 1)]
        image = random_image(label, noise=0)
        image.show()
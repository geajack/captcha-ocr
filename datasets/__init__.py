from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, random

size = 500

font_paths = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Medium.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
]

fonts = [ImageFont.truetype(path, size=size) for path in font_paths]


def pointillize(source):
    w, h = source.width, source.height
    output = Image.new("RGB", (w, h), color=(255, 255, 255))

    artist = ImageDraw.Draw(output)

    density = 35 / (50 * 50)
    n_points = int(density * w * h)

    r = 2
    for _ in range(n_points):
        x = int(random() * w)
        y = int(random() * h)
        pixel = source.getpixel((x, y))
        red = pixel[0]
        if red < 50:
            p = 1
            value = 0
        else:
            p = 0.15
            a = 50
            b = 200
            value = a + random() * (b - a)
        
        if random() < p:
            value = int(value)
            artist.ellipse((x - r, y - r, x + r, y + r), fill=(value, value, value))

    return output
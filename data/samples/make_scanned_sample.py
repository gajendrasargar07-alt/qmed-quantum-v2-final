"""Render one of the text reports as a slightly noisy scanned PNG to demo OCR."""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "prostate_report.txt")
OUT  = os.path.join(HERE, "prostate_report_scan.png")

def main():
    with open(SRC) as fp: text = fp.read()
    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    y = 30
    for line in text.splitlines():
        d.text((40, y), line, fill=(15, 15, 15), font=font)
        y += 26
    # add a faint noise / rotation to simulate a real scan
    img = img.rotate(-0.6, resample=Image.BICUBIC, fillcolor="white")
    # subtle blur to simulate scanner
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    # add speckle noise
    px = img.load()
    random.seed(3)
    for _ in range(2500):
        x = random.randint(0, W-1); yy = random.randint(0, H-1)
        v = random.randint(180, 235)
        px[x, yy] = (v, v, v)
    img.save(OUT, dpi=(200, 200))
    print("wrote", OUT, img.size)

if __name__ == "__main__":
    main()

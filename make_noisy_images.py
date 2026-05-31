import argparse
import numpy as np
from pathlib import Path
import random

import bm3d
import cv2
from skimage.util import random_noise

random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Make noisy and bM3D images")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("bm3d_path", type=Path)

    args = parser.parse_args()

    args.output.mkdir(exist_ok=False)
    args.bm3d_path.mkdir(exist_ok=False)

    for image_path in args.input.glob("*.png"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        sigma = np.round((random.random() * 0.09 + 0.01), 4)

        noisy_image = random_noise((image / 255), "gaussian", var=sigma**2)
        bm3d_image = bm3d.bm3d(noisy_image, sigma).clip(0, 1)

        noisy_image *= 255
        noisy_image = noisy_image.astype(np.uint8)
        
        bm3d_image *= 255
        bm3d_image = bm3d_image.astype(np.uint8)
        
        cv2.imwrite(str(args.output / f"{image_path.stem}-{int(sigma*(10**4))}.png"), noisy_image)
        cv2.imwrite(str(args.bm3d_path / f"{image_path.stem}-{int(sigma*(10**4))}.png"), bm3d_image)


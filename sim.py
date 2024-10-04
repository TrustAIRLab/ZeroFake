import cv2
import numpy as np
from pathlib import Path
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Process images for DDIM.")
parser.add_argument("--orginal", type=str, help="Path to the folder containing images.")
parser.add_argument("--reconstruct", type=str, help="Path to the output folder for saving images.")
args = parser.parse_args()


folder1_path = args.orginal

folder2_path = args.reconstruct

output_file = "testfake.txt"

with open(output_file, "w") as f:
    f.write("Image Filename\tPixel Similarity\n")
    index = 0
    for image_path1, image_path2 in zip(natsorted(Path(folder1_path).glob("*.*")), natsorted(Path(folder2_path).glob("*.*"))):

        image1 = cv2.imread(str(image_path1))
        image2 = cv2.imread(str(image_path2))
        image1 = cv2.resize(image1, (512, 512))
        image2 = cv2.resize(image2, (512, 512))
        
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        ssim_score = ssim(gray1, gray2)
        f.write(f"{image_path1.name}\t{ssim_score}\n")
        print("Mean pixel difference for", image_path1.name, ":", ssim_score)
        index+=1

    print("Results saved to", output_file)

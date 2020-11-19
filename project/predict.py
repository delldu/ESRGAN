"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:48:28 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import enable_amp, get_model, model_device, model_load

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/ImageZoom.pth", help="checkpint file")
    parser.add_argument('--input', type=str,
                        default="dataset/predict/LR/*.png", help="input image")
    parser.add_argument('--output', type=str,
                        default="dataset/predict/HR", help="output directory")

    args = parser.parse_args()

    model = get_model()
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    model.eval()

    enable_amp(model)

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor.clamp_(0, 1.0)
        output_tensor = output_tensor.squeeze(0)

        toimage(output_tensor.cpu()).save(
            "{}/{}".format(args.output, os.path.basename(filename)))

        del input_tensor, output_tensor
        torch.cuda.empty_cache()

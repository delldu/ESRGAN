"""Model test."""
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
import os
import argparse
import torch
from data import get_data
from model import get_model, model_load, valid_epoch, model_setenv

if __name__ == "__main__":
    """Test model."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/ImageZoom.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # get model
    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')

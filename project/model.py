"""Create model."""
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

import functools
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pdb

# The following comes from https://github.com/xinntao/ESRGAN
# Thanks a lot.


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        del x1, x2, x3, x4
        torch.cuda.empty_cache() 

        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = out * 0.2 + x
        del x
        torch.cuda.empty_cache() 

        return out

class ImageZoomModel(nn.Module):
    """ImageZoom Model."""

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        """Init model."""
        super(ImageZoomModel, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward."""
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        del x, trunk
        torch.cuda.empty_cache() 

        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        del fea
        torch.cuda.empty_cache()
        return out

def PSNR(img1, img2):
    """PSNR."""
    difference = (1.*img1-img2)**2
    mse = torch.sqrt(torch.mean(difference)) + 0.000001
    return 20*torch.log10(1./mse)

def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)


def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = ImageZoomModel()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model():
    """Create model."""
    model = ImageZoomModel(3, 3, 64, 23, gc=32)
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()
    model.train()
    criterion = nn.L1Loss()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # pdb.set_trace()

            loss = criterion(predicts, targets)
            loss_value = loss.item()

            del images, targets, predicts
            torch.cuda.empty_cache()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            loss_value = PSNR(targets, predicts)

            del images, targets, predicts
            torch.cuda.empty_cache()


            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = os.environ["DEVICE"]

    model = get_model()
    model.eval()
    model = model.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")
    print(model)

    for i in tqdm(range(10)):
        input = torch.randn(2, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        del input, output

if __name__ == '__main__':
    """Test model ..."""

    # model_export()
    infer_perform()

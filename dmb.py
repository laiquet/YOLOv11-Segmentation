from utils import (yaml_load)
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os

ROOT = os.getcwd()

def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        try:
            return yaml_load(data)["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
    """

    @torch.no_grad()
    def __init__(
        self,
        weights=None,
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=None,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt = True
        fp16 &= pt or nn_module  # FP16
        stride = 32  # default stride
        end2end = False  # default end2end
        model, metadata, task = None, None, None

        # In-memory PyTorch model
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from utils import attempt_load

            model = attempt_load(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        # Disable gradients
        if pt:
            for p in self.model.parameters():
                p.requires_grad = False

        # Explicitly assign needed attributes
        self.device = device
        self.fp16 = fp16
        self.pt = pt
        self.nn_module = nn_module
        self.stride = stride
        self.names = names
        self.fuse = fuse

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        if isinstance(y, (list, tuple)):
                if len(self.names) == 999 and len(y) == 2:  # segments and names not defined
                    nc = y[0].shape[1] - y[1].shape[1] - 4
                    self.names = {i: f"class{i}" for i in range(nc)}
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)
        warmup_types = self.pt, self.nn_module
        if any(warmup_types) and self.device.type != "cpu":
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p=None):
        """
        Determines model type from the file suffix.

        Args:
            p (str): path to the model file. Defaults to path/to/model.pt

        Returns:
            (list): Boolean list indicating which format the model is in.
        """
        sf = [".pt", ".torchscript", ".onnx", ".engine", ".tflite", ".pb",
              ".saved_model", ".mlmodel", ".xml", "_edgetpu.tflite", "_ncnn",
              "_paddle_model", ".triton"]
        name = Path(p).name if p else ""
        types = [s in name for s in sf]
        return types
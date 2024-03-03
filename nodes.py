
import torch

import comfy.utils
import comfy.model_management

import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from .utils.utils import *

script_directory = os.path.dirname(os.path.abspath(__file__))

class DSINE_normals:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "images": ("IMAGE", ),
            "fov": ("INT", {"default": 60, "min": 0, "max": 360}),
            "iterations": ("INT", {"default": 5, "min": 1, "max": 1024}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            
            },
            "optional": {
            "intrinsics_string": ("STRING", {"default": ""}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"

    CATEGORY = "DSINE"

    def process(self, images, fov, iterations, keep_model_loaded, intrinsics_string=""):
        batch_size = images.shape[0]
        device = comfy.model_management.get_torch_device()
        
        checkpoint_path = os.path.join(script_directory, f"checkpoints/dsine.pt")

        if not hasattr(self, "model") or self.model is None:
            from .models.dsine import DSINE
            self.model = DSINE().to(device)
            self.model.pixel_coords = self.model.pixel_coords.to(device)
            self.model = load_checkpoint(checkpoint_path, self.model)
            self.model.eval()
        
        self.model.num_iter = iterations
        images = images.permute(0, 3, 1, 2).to(device)
        _, _, orig_H, orig_W = images.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = pad_input(orig_H, orig_W)
        images = F.pad(images, (l, r, t, b), mode="constant", value=0.0)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)

        if intrinsics_string is not "":    #it should contain the values of fx, fy, cx, cy
            intrins_ = intrinsics_string.split(',')
            intrins_ = [float(i) for i in intrins_]
            fx, fy, cx, cy = intrins_

            intrins = torch.tensor([
                [fx, 0,cx],
                [ 0,fy,cy],
                [ 0, 0, 1]
            ], dtype=torch.float32, device=device).unsqueeze(0)

        else:
            # NOTE: if intrins is not given, we just assume that the principal point is at the center
            intrins = get_intrins_from_fov(new_fov=fov, H=orig_H, W=orig_W, device=device).unsqueeze(0)

        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t
        intrins = intrins.repeat(batch_size, 1, 1)
        
        pred_norm = self.model(images, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]
        pred_norm = pred_norm.permute(0, 2, 3, 1).cpu()
        pred_norm = torch.clip(pred_norm, 0, 1)
        if not keep_model_loaded:
            self.model = None
            comfy.model_management.soft_empty_cache()
        return pred_norm,
    
        

        


NODE_CLASS_MAPPINGS = {
    "DSINE_normals": DSINE_normals,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DSINE_normals": "DSINE_normals",
}
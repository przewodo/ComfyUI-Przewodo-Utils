import torch
import os
import glob
from PIL import Image, ImageOps
import numpy as np
import folder_paths

class BatchImagesFromPath:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "path": ("STRING", ("STRING", {"default": "",})),
                "pattern": ("STRING", {"default": "",}),
            },
        }

    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("image",)
    
    def run(self, path, pattern):
        
        path, _, _, _, _ = folder_paths.get_save_image_path("", path, 0, 0)

        print(path)

        if os.path.isabs(path):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        if not os.path.exists(path):
            return (None, )
        
        self.image_paths = []
        self.load_images(path, pattern)
        batch_images = []

        for image_path in self.image_paths:
            if not os.path.exists(image_path):
                continue

            batch_images.append(self.get_image(image_path))

        return (batch_images,)

    @classmethod
    def VALIDATE_INPUTS(self, path):
        if not folder_paths.exists_annotated_filepath(path):
            return "Path does not exist: {}".format(path)

        return True
    
    def load_images(self, folder_path, pattern):
        for file_name in glob.glob(os.path.join(glob.escape(folder_path), pattern), recursive=False):
            abs_file_path = os.path.abspath(file_name)
            self.image_paths.append(abs_file_path)

    def get_image(self, filepath):
        i = Image.open(filepath)
        i = ImageOps.exif_transpose(i)
        return torch.from_numpy(np.array(i).astype(np.float32) / 255.0).unsqueeze(0)
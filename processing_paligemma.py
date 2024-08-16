from typing import List, Tuple
import torch
import numpy as np
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image, size: Tuple[int, int], resample = None, reducing_gap = None):
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(image: np.ndarray, scale, dtype = np.float32):
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image: np.ndarray, mean, std):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean = None,
    image_std = None
):
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]
    # rescale in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # normalize
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # H, W, C -> C, H, W
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int) -> None:
        super().__init__()
        
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # special tokens for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # special tokens for object segmentation 
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        self.add_bos_token = False
        self.add_eos_token = False
        self.tokenizer = tokenizer
    
    def __call__(
        self, 
        text: List[str], 
        images: List[Image.Image], 
        padding: str = "longest", 
        truncation: bool = True
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, "image should be 1 and text also should be one"
        
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor= 1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        # B, C, H, W
        pixel_values = np.stack(pixel_values, axis=0)
        
        pixel_values = torch.tensor(pixel_values)
        
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )
        
        return_data = {
            "pixel_values": pixel_values, **inputs
        }
        
        return return_data

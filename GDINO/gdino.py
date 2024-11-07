'''
Author: hyuRen
Date: 2024-11-06 21:07:30
LastEditors: hyuRen
LastEditTime: 2024-11-06 21:33:15
Description: 
'''
import sys
sys.path.append('/home/rhy/pythonCodes/bestman_elephant_v2')
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from GDINO.utils import get_device_type

device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self, model_dir='./GDINO/gdino_model'):
        self.build_model(model_dir)

    def build_model(self, model_dir):
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(
            DEVICE
        )

    def predict(
        self,
        pil_images: list[Image.Image],
        text_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        for i, prompt in enumerate(text_prompt):
            if prompt[-1] != ".":
                text_prompt[i] += "."
        inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        return results
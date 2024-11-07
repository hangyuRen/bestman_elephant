'''
Author: hyuRen
Date: 2024-11-06 21:16:59
LastEditors: hyuRen
LastEditTime: 2024-11-06 21:37:11
Description: 
'''
from GDINO.gdino import GDINO, DEVICE
from PIL import Image
import torch

from FastSAM.fastsam import FastSAM, FastSAMPrompt


gdino_model = GDINO(model_dir='./GDINO/gdino_model')

IMAGE_PATH = 'FastSAM/images/cat.jpg'
image = Image.open(IMAGE_PATH).convert('RGB')

out = gdino_model.predict(
    [image],
    ["cat."],
    0.3,
    0.25,
)
print(out)

if(len(out[0]['labels']) == 0):
    raise RuntimeError("no object detected")

scores = out[0]['scores']
highest_score_index = torch.argmax(scores).item() 

# draw = ImageDraw.Draw(image)

box_list = out[0]['boxes'][highest_score_index].tolist()
# draw.rectangle(box_list, outline='blue', width=3)
# image.show()

box_list = [int(x) for x in box_list]
# print(box_list)
fastsam_model = FastSAM('FastSAM/fastsam/weights/FastSAM-x.pt')

everything_results = fastsam_model(image, device=DEVICE, imgsz=image.size, retina_masks=True, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)

ann = prompt_process.box_prompt(bboxes=[box_list])

prompt_process.plot(annotations=ann,output_path='test.jpg',)

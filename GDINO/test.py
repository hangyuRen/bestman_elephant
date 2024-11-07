'''
Author: hyuRen
Date: 2024-11-06 21:30:53
LastEditors: hyuRen
LastEditTime: 2024-11-06 21:32:09
Description: 
'''
from gdino import GDINO
from PIL import Image, ImageDraw
import torch
from GDINO.utils import get_box_center

gdino = GDINO(model_dir='./GDINO/gdino_model')

image = Image.open("FastSAM/images/cat.jpg").convert('RGB')
out = gdino.predict(
    [image],
    ["cat."],
    0.3,
    0.25,
)
scores = out[0]['scores']
highest_score_index = torch.argmax(scores).item() 
draw = ImageDraw.Draw(image)
# for result in out:
#     for box in result['boxes']:
#         draw.rectangle(box.tolist(), outline='blue', width=3)
# image.show()
box_list = out[0]['boxes'][highest_score_index].tolist()
center_x, center_y = get_box_center(box_list)
draw.rectangle(box_list, outline='blue', width=3)
draw.point((center_x, center_y), fill='red')
image.show()
print(out)

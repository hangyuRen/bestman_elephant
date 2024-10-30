import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils import get_device_type, get_box_center

device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self, model_dir='./gdino/gdino_model'):
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


if __name__ == "__main__":
    gdino = GDINO(model_dir='./gdino/gdino_model')
    # gdino.build_model()
    # out = gdino.predict(
    #     [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
    #     ["wheel", "wheel"],
    #     0.3,
    #     0.25,
    # )
    image = Image.open("./Asset/images/cmp1.jpg").convert('RGB')
    out = gdino.predict(
        [image],
        ["red."],
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

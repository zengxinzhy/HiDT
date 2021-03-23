from hidt.utils.preprocessing import __scale_shorter
import sys
import torch
import coremltools as ct

from PIL import Image
from tqdm import tqdm
from torch import nn
import numpy as np
from torchvision import transforms

from grid import Grid
from hidtcore import HiDT
from enhancer import Enhancer
from ops import inference_size

sys.path.append('./HiDT')
# the network has been trained to do inference in 256px, any higher value might lead to artifacts
image_path = './images/daytime/content/1.jpg'
styles_path = './styles.txt'


class CoreML(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidt = HiDT()
        self.grid = Grid()
        self.enhander = Enhancer()

    def forward(self, content, style_to_transfer):
        print(content.shape)
        content = self.grid(content)
        print(content.shape)
        content = self.hidt(content, style_to_transfer)
        print(content.shape)
        content = self.enhander(content)
        print(content.shape)
        return content


with open(styles_path) as f:
    styles = f.read()
styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(
    ',')[1][1:-1].split(' ')]) for style in styles.split('\n')[:-1]}
image = Image.open(image_path)


# Select the style, or define any vector you want
style_to_transfer = styles['sunset_hard_harder'].to('cpu')
image = transforms.Compose([
    transforms.Lambda(lambda img: __scale_shorter(img,
                                                  inference_size * 4, Image.BICUBIC)),
    transforms.ToTensor(),
])(image)

with torch.no_grad():
    model = CoreML()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.hidt.style_transformer.trainer.eval()
    # hidt.style_transformer.trainer.gen.decoder.set_style(style_to_transfer)
    for param in model.hidt.style_transformer.trainer.parameters():
        param.requires_grad = False

transferred = model(image, style_to_transfer)
transferred = transforms.ToPILImage()(transferred[0])
transferred.save("transferred.jpg")

# traced_model.save("~/hidt.pt")

traced_model = torch.jit.trace(
    model, (image, style_to_transfer), check_trace=False)

mlmodel = ct.convert(model=traced_model, inputs=[
    ct.ImageType(name="image", shape=ct.Shape(image.shape), scale=1/255.0),
    ct.TensorType(name="style", shape=ct.Shape(style_to_transfer.shape))
])
mlmodel.save("~/hidt.mlmodel")

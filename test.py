import torch
import sys
from PIL import Image
from torchvision import transforms
from grid import Grid
from enhancer import Enhancer
from hidtcore import HiDT
from ops import inference_size
from hidt.utils.preprocessing import __scale_shorter


sys.path.append('./HiDT')
# the network has been trained to do inference in 256px, any higher value might lead to artifacts
image_path = './images/daytime/content/1.jpg'
styles_path = './styles.txt'
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

grid = Grid()
hidt = HiDT()
enhancer = Enhancer()
data = grid(image)
transferred = hidt(data, style_to_transfer)
transferred = transferred.repeat(16, 1, 1, 1)
transferred = enhancer(transferred)
print(transferred.shape)
transferred = transforms.ToPILImage()(transferred[0])
transferred.save("test.jpg")

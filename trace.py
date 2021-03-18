import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from torchvision import transforms
from hidt.utils.preprocessing import __scale_shorter
from PIL import Image


class Trace(nn.Module):
    def __init__(self):
        super(Trace, self).__init__()

    def forward(self, image, style):
        return image * style.view(1, 3, 1, 1)


inference_size = 256
device = 'cpu'
image_path = './original.jpg'

# model = NeuralNetWithLoss(input_size, hidden_size, num_classes)
model = Trace()
model.eval()
image = Image.open(image_path)


# Select the style, or define any vector you want
image = transforms.Compose([
    transforms.ToTensor(),
])(image)
style = torch.ones(3) * 0.5

transferred = model(image, style)
print(transferred.shape)
for i in range(3):
    for j in range(transferred.shape[2]):
        print(transferred[0, i, j, :])
        pass
transferred = transforms.ToPILImage()(transferred[0])
transferred.save("trace.jpg")


model = torch.jit.trace(model, (image, style))

mlmodel = ct.convert(model=model, inputs=[
    ct.ImageType(name="image", shape=ct.Shape(image.shape), scale=1/255.0),
    ct.TensorType(name="style", shape=ct.Shape(style.shape))
])
mlmodel.save("~/trace.mlmodel")

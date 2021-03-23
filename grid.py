import torch
import sys
import coremltools as ct


class Grid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content):
        [c, h, w] = content.shape
        h = h // 4
        w = w // 4
        return content.reshape(c, h * 4, w, 4).transpose(2, 3).reshape(c, h, 4*4, w).transpose(1, 2).reshape(
            3, 16, h, w).transpose(0, 1).reshape(4, 4, c, h, w).transpose(0, 1).reshape(16, c, h, w)


if __name__ == '__main__':
    image = torch.zeros(3, 1024, 1808)
    model = Grid()
    traced_model = torch.jit.trace(
        model, (image), check_trace=False)
    mlmodel = ct.convert(model=traced_model, inputs=[
        ct.ImageType(name="image", shape=ct.Shape(
            shape=image.shape), scale=1/255.0),
    ])
    mlmodel.save("~/grid.mlmodel")

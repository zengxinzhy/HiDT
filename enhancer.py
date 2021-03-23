import torch
import sys
from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
import coremltools as ct
sys.path.append('./HiDT')


class Enhancer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        enhancer_weights = './trained_models/enhancer/enhancer.pth'
        self.enhancer = RRDBNet(
            in_nc=48, out_nc=3, nf=64, nb=5, gc=32).to('cpu')
        self.enhancer.load_state_dict(torch.load(enhancer_weights))

    def forward(self, content):
        # n, c, h, w = content.shape
        # transferred = content.view(1, -1, h, w)
        transferred = self.enhancer(transferred)
        transferred = (transferred.clamp(-1.0, 1.0) + 1.) / 2.
        return transferred


if __name__ == '__main__':
    data = torch.zeros(1, 48, 256, 452)
    model = Enhancer()

    traced_model = torch.jit.trace(
        model, (data), check_trace=False)
    mlmodel = ct.convert(model=traced_model, inputs=[
        ct.TensorType(name="data", shape=ct.Shape(shape=data.shape)),
    ])
    mlmodel.save("~/enhander.mlmodel")

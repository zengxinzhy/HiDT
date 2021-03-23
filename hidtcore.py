import torch
import sys
import coremltools as ct
from hidt.style_transformer import StyleTransformer
from ops import inference_size


sys.path.append('./HiDT')


class HiDT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config_path = './configs/daytime.yaml'
        gen_weights_path = './trained_models/generator/daytime.pt'
        with torch.no_grad():
            self.style_transformer = StyleTransformer(
                config_path,
                gen_weights_path,
                inference_size=inference_size,
                device='cpu')

    def forward(self, content, style_to_transfer):
        n, c, h, w = content.shape
        style_to_transfer = style_to_transfer.view(1, 1, 3, 1)
        style_to_transfer = style_to_transfer.repeat(n, 1, 1, 1)
        encoding_fn = self.style_transformer.trainer.gen.content_encoder
        content_decomposition = encoding_fn(content)
        decoder_input = {'content': content_decomposition[0],
                         'intermediate_outputs': content_decomposition[1:],
                         'style': style_to_transfer}
        transferred = self.style_transformer.trainer.gen.decode(decoder_input)[
            'images']
        return transferred.view(1, n * c, h, w)


if __name__ == '__main__':
    image = torch.zeros(1, 3, 256, 452)
    style_to_transfer = torch.zeros(3)
    model = HiDT()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.style_transformer.trainer.eval()
    for param in model.style_transformer.trainer.parameters():
        param.requires_grad = False
    # transferred = model(image, style_to_transfer)
    traced_model = torch.jit.trace(
        model, (image, style_to_transfer), check_trace=False)
    mlmodel = ct.convert(model=traced_model, inputs=[
        ct.TensorType(name="image", shape=ct.Shape(image.shape)),
        ct.TensorType(name="style", shape=ct.Shape(style_to_transfer.shape))
    ])
    mlmodel.save("~/hidtcore.mlmodel")

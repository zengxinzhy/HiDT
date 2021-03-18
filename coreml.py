from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing
from hidt.utils.preprocessing import __scale_shorter
import sys
import torch
import coremltools as ct

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import numpy as np
import inspect

# coreml conversion
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.ops.defs._op_reqs import *
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
)
from coremltools.converters.mil.backend.nn.mil_to_nn_mapping_registry import MIL_TO_NN_MAPPING_REGISTRY, register_mil_to_nn_mapping
from coremltools.converters.mil.backend.nn.op_mapping import make_input
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
sys.path.append('./HiDT')
config_path = './configs/daytime.yaml'
gen_weights_path = './trained_models/generator/daytime.pt'
# the network has been trained to do inference in 256px, any higher value might lead to artifacts
inference_size = 256
device = 'cpu'
image_path = './original.jpg'
styles_path = './styles.txt'
enhancer_weights = './trained_models/enhancer/enhancer.pth'

# del SSAOpRegistry.ops["core"]["batch_norm"]
# del MIL_TO_NN_MAPPING_REGISTRY["batch_norm"]
# del _TORCH_OPS_REGISTRY["batch_norm"]

# @register_op(doc_str="")
# class batch_norm(Operation):
#     input_spec = InputSpec(
#         x=TensorInputType(),
#         gamma=TensorInputType(),
#         beta=TensorInputType(),
#         mean=TensorInputType(const=True),
#         variance=TensorInputType(const=True),
#         epsilon=FloatInputType(const=True, optional=True),
#     )

#     def default_inputs(self):
#         return DefaultInputs(
#             epsilon=1e-5,
#         )

#     def __init__(self, **kwargs):
#         super(batch_norm, self).__init__(**kwargs)

#     def type_inference(self):
#         return types.tensor(types.fp32, tuple(self.x.shape))


# def batch_norm_registry(const_context, builder, op):
#     channels = op.x.shape[1]
#     x_name = make_input(const_context, builder, op.x)
#     gamma_name = make_input(const_context, builder, op.gamma)
#     beta_name = make_input(const_context, builder, op.beta)
#     out_name = op.outputs[0].name
#     print(op.x, op.gamma, op.beta)

#     # Set the parameters
#     spec_layer = builder._add_generic_layer(
#         op.name, [x_name, gamma_name, beta_name], [out_name])
#     spec_layer_params = spec_layer.batchnorm
#     spec_layer_params.channels = channels
#     spec_layer_params.epsilon = op.epsilon.val
#     spec_layer_params.computeMeanVar = False
#     spec_layer_params.instanceNormalization = False
#     print(spec_layer_params)


# batch_norm_registry.__name__ = "batch_norm"
# register_mil_to_nn_mapping(batch_norm_registry)


@register_torch_op
def type_as(context, node):
    inputs = _get_inputs(context, node)
    context.add(mb.cast(x=inputs[0], dtype='fp32'), node.name)


class HiDT(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            style_transformer = StyleTransformer(config_path,
                                                 gen_weights_path,
                                                 inference_size=inference_size,
                                                 device=device)
        self.transformer = style_transformer
        self.transform = transforms.Compose([
            # transforms.Lambda(lambda x: x + torch.rand_like(x) / 255),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.style_transformer = style_transformer

    def forward(self, content, style_to_transfer):
        style_to_transfer = style_to_transfer.view(1, 1, 3, 1).to(device)
        content = self.transform(content)
        content = content.view(1, *content.shape)
        encoding_fn = self.style_transformer.trainer.gen.content_encoder
        content_decomposition = encoding_fn(content)
        decoder_input = {'content': content_decomposition[0],
                         'intermediate_outputs': content_decomposition[1:],
                         'style': style_to_transfer}
        transferred = self.style_transformer.trainer.gen.decode(decoder_input)[
            'images']
        transferred = (transferred.clamp(-1.0, 1.0) + 1.) / 2.
        return transferred


with open(styles_path) as f:
    styles = f.read()
styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(
    ',')[1][1:-1].split(' ')]) for style in styles.split('\n')[:-1]}
image = Image.open(image_path)

# Select the style, or define any vector you want
style_to_transfer = styles['sunset_hard_harder']
image = transforms.Compose([
    transforms.ToTensor(),
])(image)
transforms.ToPILImage()(image).save("original.jpg")
with torch.no_grad():
    hidt = HiDT()
    hidt.eval()
    for param in hidt.parameters():
        param.requires_grad = False
    hidt.style_transformer.trainer.eval()
    # hidt.style_transformer.trainer.gen.decoder.set_style(style_to_transfer)
    for param in hidt.style_transformer.trainer.parameters():
        param.requires_grad = False

transferred = hidt(image, style_to_transfer)
print(transferred.shape)
transferred = transforms.ToPILImage()(transferred[0])
transferred.save("transferred.jpg")

# traced_model.save("~/hidt.pt")

traced_model = torch.jit.trace(
    hidt, (image, style_to_transfer), check_trace=False)

mlmodel = ct.convert(model=traced_model, inputs=[
    ct.ImageType(name="image", shape=ct.Shape(image.shape), scale=1/255.0),
    ct.TensorType(name="style", shape=ct.Shape(style_to_transfer.shape))
])
mlmodel.save("~/hidt.mlmodel")

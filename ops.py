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

inference_size = 256


@register_torch_op
def type_as(context, node):
    inputs = _get_inputs(context, node)
    context.add(mb.cast(x=inputs[0], dtype='fp32'), node.name)




from minimind.modeling.architectures import Architecture, register_architecture


@register_architecture
class BaymaxTransformer(Architecture):
    "Create the world-best multimodal emotional Baymax"

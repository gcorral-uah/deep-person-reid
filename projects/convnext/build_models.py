from .convnext import ConvNeXt
import torch


def build_convnext(num_classes=1000, pretrained=True, **kwargs):
    model = ConvNeXt(
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs
    )
    if pretrained:
        weights_url = (
            "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth"
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url=weights_url, map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])

    return model

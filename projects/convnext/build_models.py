import convnext
import torch


def build_convnext(num_classes=1000, pretrained=True, **kwargs):
    model = convnext.ConvNeXt(
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
        # We don't want to load the pretrained weights of the head of the
        # network, as the number of output classes differ, so we delete those
        # layers from the dict. Additionally we have to pass strict = False to
        # load_state_dict, so that we can avoid an error when the
        # 'head.weight/bias' keys are not present in the pretrained dict.
        weights = checkpoint["model"]
        del weights['head.weight']
        del weights['head.bias']
        model.load_state_dict(weights, strict=False)

    return model

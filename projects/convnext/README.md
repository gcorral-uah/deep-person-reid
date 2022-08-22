# Introduction
This repository includes the code to train the
[convnext](https://github.com/facebookresearch/ConvNeXt) model for person
reidentification tasks.

We have also developed a model based on convnext and OSNet ideas to try to
improve the model performance. In particular we adopt the idea of using
multiple convolutional blocks, with different receptive fields, that we later
combine with a gating network to learn different characteristics dynamically.

The convolutional blocks with greater receptive fields are based on stacked
convolutions of the same block type (with a kernel size of 7, a stride of 1 and
a padding of 3), which generates a $(3t + 1) \cdot (3t + 1)$ [^1] receptive
field size.

[^1]: I am not sure of this equation

import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from module.mobilent_v2 import MobileNetV2, InvertedResidual
from module.ssd import SSD, GraphPath
from utils.predictor import Predictor
from utils.argument import _argument
from model.config import mb_ssd_lite_f38_config as config
from torchscope import scope


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=True):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mb_ssd_lite_f38(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=True, is_test=False,
                           device="cuda"):
    anchors = [2, 2, 2, 2]
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm,
                           onnx_compatible=onnx_compatible).features

    source_layer_indexes = [GraphPath(7, 'conv', 3), GraphPath(14, 'conv', 3), 19, ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(192 * width_mult), out_channels=anchors[0] * 4, kernel_size=3, padding=1,
                        onnx_compatible=onnx_compatible),
        SeperableConv2d(in_channels=576, out_channels=anchors[1] * 4, kernel_size=3, padding=1,
                        onnx_compatible=onnx_compatible),
        SeperableConv2d(in_channels=1280, out_channels=anchors[2] * 4, kernel_size=3, padding=1,
                        onnx_compatible=onnx_compatible),
        SeperableConv2d(in_channels=512, out_channels=anchors[3] * 4, kernel_size=3, padding=1,
                        onnx_compatible=onnx_compatible),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(192 * width_mult), out_channels=anchors[0] * num_classes, kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=576, out_channels=anchors[1] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=anchors[2] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=anchors[3] * num_classes, kernel_size=3, padding=1),
    ])
    landmark_headers = ModuleList([
        SeperableConv2d(in_channels=round(192 * width_mult), out_channels=anchors[0] * 10, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=576, out_channels=anchors[1] * 10, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=anchors[2] * 10, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=anchors[3] * 10, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, landmark_headers, is_test=is_test, config=config,
               device=device)


def create_mb_ssd_lite_f38_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor


if __name__ == "__main__":
    model = create_mb_ssd_lite_f38(num_classes=2)
    model.eval()
    scope(model, input_size=(3, 300, 300))
    ##################export###############
    output_onnx = 'face_ssd_landmark.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["conf0", "loc0", "landmark0"]
    inputs = torch.randn(1, 3, 300, 300)
    torch_out = torch.onnx._export(model,
                                   inputs,
                                   output_onnx,
                                   verbose=True,
                                   input_names=input_names,
                                   output_names=output_names,
                                   example_outputs=True,  # to show sample output dimension
                                   keep_initializers_as_inputs=True,  # to avoid error _Map_base::at
                                   # opset_version=11, # need to change to 11, to deal with tensorflow fix_size input
                                   # dynamic_axes={
                                   #     "input0": [2, 3],
                                   #     "loc0": [1, 2],
                                   #     "conf0": [1, 2],
                                   #     "landmark0": [1, 2]
                                   # }
                                   )

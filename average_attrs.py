import os
from types import MethodType

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import (Deconvolution, DeepLift, DeepLiftShap, GradientShap,
                         GuidedBackprop, InputXGradient, IntegratedGradients,
                         Saliency)
from sklearn.metrics.pairwise import cosine_similarity

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from modified_classes import ModifiedFastRCNNOutputLayers, ModifiedImageList


class WrapperModel(torch.nn.Module):
      def __init__(self, modified, device):
            super().__init__()
            self.model = modified
            self.device = device

      def forward(self, input):
            outputs = self.model.inference(input, do_postprocess=False)
            acc = []
            for i in range(len(outputs)):
                  if outputs[i].shape[0] != 0:
                        acc.append(outputs[i][0].unsqueeze(0))
                  else:
                        acc.append(torch.cat([outputs[i],
                                          torch.zeros((1, outputs[i].shape[1])).to(self.device)]))
            return torch.cat(acc)

def main(image_path, weights_path):
    device = torch.device("cuda")

    # read sample image
    img = cv2.imread(image_path)

    # resize image
    while img.shape[0] > 500:
        img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2), interpolation = cv2.INTER_AREA)

    # convert to tensor, define baseline and baseline distribution
    input_   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).type(torch.cuda.FloatTensor)
    baseline = torch.zeros(input_.shape).to(device).type(torch.cuda.FloatTensor)
    baseline_dist = torch.randn(5, input_.shape[1], input_.shape[2], input_.shape[3]).to(device) * 0.001

    model_path = weights_path

    # load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path

    model = build_model(cfg).to(device).eval()
    model.roi_heads.box_predictor = ModifiedFastRCNNOutputLayers(model.roi_heads.box_predictor)
    model.preprocess_image = MethodType(new_preprocess_image, model)
    model.roi_heads.forward_with_given_boxes = MethodType(lambda self, x, y: y, model)

    modified = model

    DetectionCheckpointer(modified).load(cfg.MODEL.WEIGHTS)
    modified.roi_heads.box_predictor.class_scores_only = True
    modified.to(device)

    wrapper = WrapperModel(modified, device)
    pred_class = 0

    # Integrated Gradients
    ig = IntegratedGradients(wrapper)
    ig_attributions, _ = ig.attribute(input_,
                                        target=pred_class,
                                        return_convergence_delta=True)

    # Gradient SHAP
    gs = GradientShap(wrapper)
    gs_attributions, _ = gs.attribute(input_,
                                        stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                        target=pred_class, 
                                        return_convergence_delta=True)


    # Deep Lift
    dl = DeepLift(wrapper)
    dl_attributions, _ = dl.attribute(input_, baseline, target=pred_class, return_convergence_delta=True)

    # DeepLiftShap
    dls = DeepLiftShap(wrapper)
    dla_attributions, _ = dls.attribute(input_.float(), baseline_dist, target=pred_class, return_convergence_delta=True)

    # Saliency
    saliency = Saliency(wrapper)
    saliency_attributions = saliency.attribute(input_, target=pred_class)

    # InputXGradient
    inputxgradient = InputXGradient(wrapper)
    inputxgradient_attributions = inputxgradient.attribute(input_, target=pred_class)

    # Deconvolution
    deconv = Deconvolution(wrapper)
    deconv_attributions = deconv.attribute(input_, target=pred_class)

    # Guided Backprop
    gbp = GuidedBackprop(wrapper)
    gbp_attributions = gbp.attribute(input_, target=pred_class)

    attrs = [ig_attributions, gs_attributions, 
            dl_attributions, dla_attributions, 
            saliency_attributions, inputxgradient_attributions, 
            deconv_attributions, gbp_attributions]
    processed_attrs = [process_attr(attr).reshape((-1)) for attr in attrs]
    sim_mat = cosine_similarity(np.stack(processed_attrs))

    print("Average cosine similarity between attributions:", (np.sum(sim_mat) - len(attrs))/ (len(attrs)**2 - len(attrs)))


def new_preprocess_image(self, batched_inputs: torch.Tensor):
    """
    Normalize, pad and batch the input images.
    """
    images = [x.to(self.device) for x in batched_inputs]
    images = [(x - self.pixel_mean) / self.pixel_std for x in images]
    images = ModifiedImageList.from_tensors(images, self.backbone.size_divisibility) # Extend ImageList to new object
    return images


def process_attr(attributions):
    # C, H, W -> H, W, C
    attributions = attributions[0].permute(1,2,0).detach().cpu().numpy()

    # flattern to 1D
    attributions = np.sum(np.abs(attributions), axis=-1)

    # normalise attributions to [0,1]
    attributions -= np.min(attributions)
    attributions /= np.max(attributions)

    return attributions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="dataset/night/20201201_000505.jpg")
    parser.add_argument("-w", "--weights_path", type=str, default="assets/frcnn-100epochs/frcnn-100epochs.pt")
    args = parser.parse_args()
    main(image_path=args.image_path, weights_path=args.weights_path)

from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2
from captum.attr import (IntegratedGradients, GradientShap, 
                         DeepLift, DeepLiftShap,
                         Saliency, InputXGradient,
                         Deconvolution, GuidedBackprop, GuidedGradCam,
                         FeatureAblation, Occlusion)
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

def main():
    device = torch.device("cuda")

    # read sample image
    img = cv2.imread('dataset/night/20201201_000505.jpg')

    # resize image
    while img.shape[0] > 500:
        img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2), interpolation = cv2.INTER_AREA)

    # convert to tensor, define baseline and baseline distribution
    input_   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).type(torch.cuda.FloatTensor)
    baseline = torch.zeros(input_.shape).to(device).type(torch.cuda.FloatTensor)
    baseline_dist = torch.randn(5, input_.shape[1], input_.shape[2], input_.shape[3]).to(device) * 0.001

    # load model
    model_path = 'assets/crocodile_frcnn.pt'

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
    attributions, delta = ig.attribute(input_,
                                    target=pred_class,
                                    return_convergence_delta=True)
    print('Integrated Gradients Convergence Delta:', delta)
    save_attr_mask(attributions, img, 'IG')

    # Gradient SHAP
    gs = GradientShap(wrapper)
    attributions, delta = gs.attribute(input_,
                                    stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                    target=pred_class, 
                                    return_convergence_delta=True)

    print('GradientShap Convergence Delta:', delta)
    print('GradientShap Average Delta per example:', torch.mean(delta.reshape(input_.shape[0], -1), dim=1))
    save_attr_mask(attributions, img, 'GradientShap')

    # Deep Lift
    dl = DeepLift(wrapper)
    attributions, delta = dl.attribute(input_, baseline, target=pred_class, return_convergence_delta=True)
    print('DeepLift Convergence Delta:', delta)
    save_attr_mask(attributions, img, 'DeepLift')

    # DeepLiftShap
    dls = DeepLiftShap(wrapper)
    attributions, delta = dls.attribute(input_.float(), baseline_dist, target=pred_class, return_convergence_delta=True)
    print('DeepLiftShap Convergence Delta:', delta)
    print('Deep Lift SHAP Average delta per example:', torch.mean(delta.reshape(input_.shape[0], -1), dim=1))
    save_attr_mask(attributions, img, 'DeepLiftShap')

    # Saliency
    saliency = Saliency(wrapper)
    attribution = saliency.attribute(input_, target=pred_class)
    save_attr_mask(attribution, img, 'Saliency')

    # InputXGradient
    inputxgradient = InputXGradient(wrapper)
    attribution = inputxgradient.attribute(input_, target=pred_class)
    save_attr_mask(attribution, img, 'InputXGradient')

    # Deconvolution
    deconv = Deconvolution(wrapper)
    attribution = deconv.attribute(input_, target=pred_class)
    save_attr_mask(attribution, img, 'Deconvolution')

    # Guided Backprop
    gbp = GuidedBackprop(wrapper)
    attribution = gbp.attribute(input_, target=pred_class)
    save_attr_mask(attribution, img, 'GuidedBackprop')

    # # GuidedGradCam
    # guided_gc = GuidedGradCam(wrapper, wrapper.model.backbone) # TODO: doesnt seem right
    # attribution = guided_gc.attribute(input_, target=pred_class, attribute_to_layer_input=True)
    # save_attr_mask(attribution, img, 'GuidedGradCam')

    # FeatureAblation
    ablator = FeatureAblation(wrapper)
    attr = ablator.attribute(input_, target=pred_class, show_progress=True)
    save_attr_mask(attr, img, 'FeatureAblation')

    # Occlusion
    ablator = Occlusion(wrapper)
    attr = ablator.attribute(input_, target=pred_class, sliding_window_shapes=(1, 3,3))
    save_attr_mask(attr, img, 'Occlusion')

def new_preprocess_image(self, batched_inputs: torch.Tensor):
      """
      Normalize, pad and batch the input images.
      """
      images = [x.to(self.device) for x in batched_inputs]
      images = [(x - self.pixel_mean) / self.pixel_std for x in images]
      images = ModifiedImageList.from_tensors(images, self.backbone.size_divisibility) # Extend ImageList to new object
      return images


def save_attr_mask(attributions, img, algo_name):
    # C, H, W -> H, W, C
    attributions = attributions[0].permute(1,2,0).detach().cpu().numpy()

    # flattern to 1D
    attributions = np.sum(np.abs(attributions), axis=-1)

    # normalise attributions
    attributions -= np.min(attributions)
    attributions /= np.max(attributions)

    _, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(attributions, cmap=plt.cm.inferno)
    axs[0, 0].axis('off')
    axs[0, 1].set_title(f'Overlay {algo_name} on Input image ')
    axs[0, 1].imshow(attributions, cmap=plt.cm.inferno)
    axs[0, 1].imshow(img, alpha=0.5)
    axs[0, 1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{algo_name}_mask.png', bbox_inches='tight')


if __name__ == "__main__":
    main()

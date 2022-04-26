from typing import List, Tuple

import torch
from detectron2.layers import batched_nms
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, Instances


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    class_scores_only: bool
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        if class_scores_only:
            class_scores: (list[Tensor]): A list of Tensors of predicted class scores for each image.
        else:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.

        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, class_scores_only
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]

    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    class_scores_only: bool
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    class_scores = scores.clone()

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    class_scores = class_scores[filter_inds[:,0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, class_scores = boxes[keep], scores[keep], filter_inds[keep], class_scores[keep]

    if class_scores_only:
        boxes = boxes.detach()
        scores = scores.detach()
        filter_inds = filter_inds.detach()

        # class_scores: tensor[N x K]
        return class_scores, filter_inds[:, 0]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.class_scores = class_scores
    result.pred_classes = filter_inds[:, 1]

    return result, filter_inds[:, 0]


class ModifiedFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self, fast_rcnn_output_layers_instance):
        self.class_scores_only = False
        super().__init__(input_shape = fast_rcnn_output_layers_instance.cls_score.in_features,
                         box2box_transform = fast_rcnn_output_layers_instance.box2box_transform,
                         num_classes = fast_rcnn_output_layers_instance.num_classes,
                         test_score_thresh = fast_rcnn_output_layers_instance.test_score_thresh,
                         test_nms_thresh = fast_rcnn_output_layers_instance.test_nms_thresh,
                         test_topk_per_image = fast_rcnn_output_layers_instance.test_topk_per_image,
                         cls_agnostic_bbox_reg = fast_rcnn_output_layers_instance.num_classes == 1,
                         smooth_l1_beta = fast_rcnn_output_layers_instance.smooth_l1_beta,
                         box_reg_loss_type = fast_rcnn_output_layers_instance.box_reg_loss_type,
                         loss_weight = fast_rcnn_output_layers_instance.loss_weight
                        )

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        # print('PREDICTIONS:', [p.shape for p in predictions])
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.class_scores_only
        )
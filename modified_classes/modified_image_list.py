from typing import List, Tuple

import torch
from detectron2.layers.wrappers import shapes_to_tensor
from detectron2.structures import ImageList
from torch.nn import functional as F


class ModifiedImageList(ImageList):
    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        super().__init__(tensor, image_sizes)

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ModifiedImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            a `ModifiedImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            print("Inside ModifiedImageList.from_tensors")
            print(len(tensors)) # N, C, H, W
            
            '''
            basically what's happening here is padding the list of all images into the same max_size
            so that we can pass a N, C, H, W tensor to the model
            technically we would not face this issue as we are alreadying passing a N, C, H, W tensor
            right from the start, this step is to compensate for allowing training to occur using
            detectron2 abstractions, i.e. a dict w 'image' as one of the keys
            '''

            # TODO: since we are copying the above code almost directly (except compensating for a for loop)
            #       maybe dont need to check for len(tensors) == 1? can merge or nah?

            images_list = []
            for i in range(len(tensors)):
                image_size = image_sizes[i]
                padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
                padded_img = F.pad(tensors[i], padding_size, value=pad_value)
                images_list.append(padded_img)
            batched_imgs = torch.stack(images_list)

        return ModifiedImageList(batched_imgs.contiguous(), image_sizes)

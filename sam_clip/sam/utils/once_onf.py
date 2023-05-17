''' Image => Mask'''
from torch.nn import functional as F
from typing import Tuple
from ..modeling import Sam
from .amg import calculate_stability_score
from sam import sam_model_registry, SamPredictor
import sys
sys.path.append("..")
from ..build_sam import sam_model_registry
from ..automatic_mask_generator import SamAutomaticMaskGenerator
import torch
import torch.nn as nn
import numpy as np

class once_onf(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(torch.int64)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
    ):
        ''''''
        upscaled_masks_list, scores_list, masks_list=[],[],[]
        # 获取图片尺寸
        height, width = 534, 800
        # 构造整张图片尺度的矩形框
        # input_box = np.array([0, 0, width, height])
        # 定义多个框的坐标和标签
        num_boxes = 4
        step_x = width // (num_boxes + 1)
        step_y = height // (num_boxes + 1)
        boxes = [
            [((i+1)*step_x - 50), ((j+1)*step_y - 50), ((i+1)*step_x + 50), ((j+1)*step_y + 50)]
            for i in range(num_boxes) for j in range(num_boxes)
        ]
        
        for input_box in boxes:
            input_box=np.array(input_box)
            point_coords = input_box.reshape(2, 2)
            point_coords=torch.from_numpy(point_coords).unsqueeze(0)
            point_labels = torch.tensor([1], dtype=torch.float).unsqueeze(0)
            
            embed_size = self.model.prompt_encoder.image_embedding_size
            mask_input_size = [4 * x for x in embed_size]
            mask_input=torch.randn(1, 1, *mask_input_size, dtype=torch.float)
            has_mask_input=torch.tensor([1], dtype=torch.float)
            orig_im_size=torch.tensor([534, 800], dtype=torch.float)
            
            ''''''
            sparse_embedding = self._embed_points(point_coords, point_labels)
            dense_embedding = self._embed_masks(mask_input, has_mask_input)

            masks, scores = self.model.mask_decoder.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding,
            )
            if self.use_stability_score:
                scores = calculate_stability_score(
                    masks, self.model.mask_threshold, self.stability_score_offset
                )

            if self.return_single_mask:
                masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

            upscaled_masks = self.mask_postprocessing(masks, orig_im_size)
        
        upscaled_masks_list.append(upscaled_masks)
        scores_list.append(scores)
        masks_list.append(masks)

        return upscaled_masks_list, scores_list, masks_list

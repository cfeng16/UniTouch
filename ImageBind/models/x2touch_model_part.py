#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import urllib
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
from collections import OrderedDict

from .helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from .multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)

from .transformer import MultiheadAttention, SimpleTransformer


ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
    TOUCH="touch"
)


class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        touch_embed_dim=1024,
        touch_num_blocks=24,
        touch_num_heads=16,
        modality_list=['vision', 'text', 'audio', 'thermal', 'depth', 'imu', 'touch']
    ):
        super().__init__()

        self.modality_list = modality_list

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
            touch_embed_dim
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
            touch_embed_dim,
            touch_num_blocks,
            touch_num_heads
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
            touch_embed_dim
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
        touch_embed_dim=1024
        
    ):

        rgbt_preprocessor, text_preprocessor, audio_preprocessor, depth_preprocessor, thermal_preprocessor, imu_preprocessor, touch_preprocessor = None, None, None, None, None, None, None

        if 'vision' in self.modality_list:
            rgbt_stem = PatchEmbedGeneric(
                proj_stem=[
                    PadIm2Video(pad_type="repeat", ntimes=2),
                    nn.Conv3d(
                        in_channels=3,
                        kernel_size=kernel_size,
                        out_channels=vision_embed_dim,
                        stride=kernel_size,
                        bias=False,
                    ),
                ]
            )
        
            rgbt_preprocessor = RGBDTPreprocessor(
                img_size=[3, video_frames, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                rgbt_stem=rgbt_stem,
                depth_stem=None,
            )

        if 'text' in self.modality_list:
            text_preprocessor = TextPreprocessor(
                context_length=77,
                vocab_size=49408,
                embed_dim=text_embed_dim,
                causal_masking=True,
            )

        if 'audio' in self.modality_list:
            audio_stem = PatchEmbedGeneric(
                proj_stem=[
                    nn.Conv2d(
                        in_channels=1,
                        kernel_size=audio_kernel_size,
                        stride=audio_stride,
                        out_channels=audio_embed_dim,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
            )
            audio_preprocessor = AudioPreprocessor(
                img_size=[1, audio_num_mel_bins, audio_target_len],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                audio_stem=audio_stem,
            )

        if 'depth' in self.modality_list:
            depth_stem = PatchEmbedGeneric(
                [
                    nn.Conv2d(
                        kernel_size=depth_kernel_size,
                        in_channels=1,
                        out_channels=depth_embed_dim,
                        stride=depth_kernel_size,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
            )

            depth_preprocessor = RGBDTPreprocessor(
                img_size=[1, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                rgbt_stem=None,
                depth_stem=depth_stem,
            )

        if 'thermal' in self.modality_list:
            thermal_stem = PatchEmbedGeneric(
                [
                    nn.Conv2d(
                        kernel_size=thermal_kernel_size,
                        in_channels=1,
                        out_channels=thermal_embed_dim,
                        stride=thermal_kernel_size,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
            )
            thermal_preprocessor = ThermalPreprocessor(
                img_size=[1, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                thermal_stem=thermal_stem,
            )

        if 'imu' in self.modality_list:
            imu_stem = PatchEmbedGeneric(
                [
                    nn.Linear(
                        in_features=48,
                        out_features=imu_embed_dim,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
            )

            imu_preprocessor = IMUPreprocessor(
                img_size=[6, 2000],
                num_cls_tokens=1,
                kernel_size=8,
                embed_dim=imu_embed_dim,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                imu_stem=imu_stem,
            )

        # 新增：
        if 'touch' in self.modality_list:
            touch_stem = PatchEmbedGeneric(
                proj_stem=[
                    PadIm2Video(pad_type="repeat", ntimes=2),
                    nn.Conv3d(
                        in_channels=3,
                        kernel_size=kernel_size,
                        out_channels=touch_embed_dim,
                        stride=kernel_size,
                        bias=False,
                    ),
                ]
            )
            touch_preprocessor = RGBDTPreprocessor(
                img_size=[3, video_frames, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                rgbt_stem=touch_stem,
                depth_stem=None,
            )


        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
            ModalityType.TOUCH: touch_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        touch_embed_dim=1024,
        touch_num_blocks=24,
        touch_num_heads=16
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        if 'vision' in self.modality_list:
            modality_trunks[ModalityType.VISION] = instantiate_trunk(
                vision_embed_dim,
                vision_num_blocks,
                vision_num_heads,
                pre_transformer_ln=True,
                add_bias_kv=False,
                drop_path=0.0,
            )
        
        if 'text' in self.modality_list:
            modality_trunks[ModalityType.TEXT] = instantiate_trunk(
                text_embed_dim,
                text_num_blocks,
                text_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=False,
                drop_path=0.0,
            )
        
        if 'audio' in self.modality_list:
            modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
                audio_embed_dim,
                audio_num_blocks,
                audio_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=audio_drop_path,
            )
        if 'depth' in self.modality_list:
            modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
                depth_embed_dim,
                depth_num_blocks,
                depth_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=depth_drop_path,
            )
        
        if 'thermal' in self.modality_list:
            modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
                thermal_embed_dim,
                thermal_num_blocks,
                thermal_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=thermal_drop_path,
            )
        
        if 'imu' in self.modality_list:
            modality_trunks[ModalityType.IMU] = instantiate_trunk(
                imu_embed_dim,
                imu_num_blocks,
                imu_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=imu_drop_path,
            )

        # 新增：
        # 不确定pre_transformer_ln=True 以及 drop_path 和 add_bias_kv
        if 'touch' in self.modality_list:
            modality_trunks[ModalityType.TOUCH] = instantiate_trunk(
                touch_embed_dim,
                touch_num_blocks,
                touch_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=0.0,
            )
        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
        touch_embed_dim
    ):
        modality_heads = {}

        if 'vision' in self.modality_list:
            modality_heads[ModalityType.VISION] = nn.Sequential(
                nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
            )

        if 'text' in self.modality_list:
            modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
                proj=nn.Sequential(
                    nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                    #nn.Linear(text_embed_dim, out_embed_dim, bias=False),
                )
            )

        if 'audio' in self.modality_list:
            modality_heads[ModalityType.AUDIO] = nn.Sequential(
                nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
            )

        if 'depth' in self.modality_list:
            modality_heads[ModalityType.DEPTH] = nn.Sequential(
                nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
            )

        if 'thermal' in self.modality_list:
            modality_heads[ModalityType.THERMAL] = nn.Sequential(
                nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
            )

        if 'imu' in self.modality_list:
            modality_heads[ModalityType.IMU] = nn.Sequential(
                nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Dropout(p=0.5),
                nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
            )

        # 新增
        if 'touch' in self.modality_list:
            modality_heads[ModalityType.TOUCH] = nn.Sequential(
                nn.LayerNorm(normalized_shape=touch_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(touch_embed_dim, out_embed_dim, bias=False),
            )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}
        
        if 'vision' in self.modality_list:
            modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        
        if 'text' in self.modality_list:
            modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
                Normalize(dim=-1), LearnableLogitScaling(learnable=True)
            )

        if 'audio' in self.modality_list:
            modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
            )
        
        if 'depth' in self.modality_list:
            modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
            )
        
        if 'thermal' in self.modality_list:
            modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
            )

        if 'imu' in self.modality_list:
            modality_postprocessors[ModalityType.IMU] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
            )

        # 新增：不确定
        if 'touch' in self.modality_list:
            modality_postprocessors[ModalityType.TOUCH] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
            )


        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs, normalize=True):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                if normalize:
                    modality_value = self.modality_postprocessors[modality_key](
                        modality_value
                    )
                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)
                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        modality_list=['vision', 'touch']
        # modality_list=['vision', 'text', 'audio', 'thermal', 'depth', 'imu', 'touch']
        
    )

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))

        # 新增:不确定
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"), strict=False)
        # print('missing key', missing_keys)
        # print('unexpected_keys', unexpected_keys)

    return model

# 新增
def x2touch(pretrained=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        modality_list=['vision', 'text', 'audio', 'thermal', 'depth', 'imu', 'touch']
    )

    if pretrained:
        # if not os.path.exists(".checkpoints/imagebind_huge.pth"):
        #     print('no checkpoint')
        #     exit()

        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))

        # 新增:不确定
        checkpoint_path = './last_new.ckpt'
        if checkpoint_path.endswith('.ckpt'):
            print('transferring ckpt to pth')
            ckpt = torch.load(checkpoint_path)
            # print(ckpt.keys())
            # print(ckpt['state_dict'].type())

            new_pth = OrderedDict()

            for key in ckpt['state_dict'].keys():
                new_key = key[6:]
                new_pth[new_key] = ckpt['state_dict'][key]

            model.load_state_dict(new_pth, strict=False)
            new_path = checkpoint_path[:-5] + '.pth'
            # torch.save(model.state_dict(), new_path)
            print('finish transferrring and loading ckpt')
        else:
            pass
            print('loading ckpt')
            print(checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path), strict=True)

    return model


def save_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    try:
        torch.save(module_dict.state_dict(),
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
        logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not save module parameters for {module_name} to {checkpoint_dir}.")


def load_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    try:
        module_dict.load_state_dict(torch.load(
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")), strict=False)
        logging.info(f"Loaded parameters for module {module_name} from {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}.")


class ImageBindEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = x2touch(pretrained=True, modality_list=['vision', 'touch', 'text'])
        self.model = x2touch(pretrained=True)
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
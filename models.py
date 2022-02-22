import numpy as np
import torch
from torch import nn
from video_dataset import VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pytorchvideo.models.vision_transformers
import torch.nn.functional as F
import torch.optim as optim
import r2plus1d
import x3d
import swin_transformer
import torchvision_resnet


embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
pool_kv_stride_adaptive = [1, 8, 8]
pool_kvq_kernel = [3, 3, 3]
def make_mvit( **kwargs):
  return pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(
      spatial_size=resize,
      temporal_size=temporal_frame,# RGB input from Kinetics
      depth=16,
      num_heads=2,
      embed_dim_mul=embed_dim_mul,
      atten_head_mul=atten_head_mul,
      pool_q_stride_size=pool_q_stride_size,
      pool_kv_stride_adaptive=pool_kv_stride_adaptive,
      pool_kvq_kernel=pool_kvq_kernel,
      head_num_classes=classes, # Kinetics has 400 classes so we need out final head to align
      **kwargs,
  )

def make_swin_transformer():
  return swin_transformer.SwinTransformer3D(

      depths=[2,2,2,2],

  )

def make_resnet3d_50():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3,
      model_depth=50,
      model_num_class=50,
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )

def make_resnet3d_18(**kwargs):
  return torchvision_resnet.r3d_18(
      pretrained=False,
      num_classes = 50,
      **kwargs

  )

def make_r2plus1d_18(**kwargs):
  return torchvision_resnet.r2plus1d_18(
      pretrained=False,
      num_classes = 50,
      **kwargs

  )

def make_r2plus1d():
  return r2plus1d.create_r2plus1d(
      input_channel=3,
      model_depth=50,
      model_num_class=50,
      norm=nn.BatchNorm3d,
      dropout_rate=0.0,
      activation=nn.ReLU,
  )




def make_slowfast():
  return pytorchvideo.models.slowfast.create_slowfast(
      model_depth=18,
      model_num_class=50,
  )
def make_x3d():
    return x3d.create_x3d(
        input_clip_length=temporal_frame,
        input_crop_size=resize,
        model_num_class=50,
    )

def make_csn():
    return pytorchvideo.models.csn.create_csn(
        model_num_class=50,
        dropout_rate=0.0,

    )

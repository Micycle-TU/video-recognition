import numpy as np
import torch
from torch import nn
from video_dataset import VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import torchvision_resnet
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pytorchvideo.models.vision_transformers
import torch.nn.functional as F
import torch.optim as optim
import r2plus1d
import x3d
import swin_transformer

batch = 6
num_seg = 8
frames_per_seg = 1
resize = 112
temporal_frame = num_seg * frames_per_seg
learning_rate = 0.000025
weight_decay = 0.05
classes = 50


embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
pool_kv_stride_adaptive = [1, 8, 8]
pool_kvq_kernel = [3, 3, 3]
def make_mvit():
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

def make_resnet3d_18():
  return torchvision_resnet.r3d_18(
      pretrained=False,
      num_classes = 50,

  )

def make_r2plus1d_18():
  return torchvision_resnet.r2plus1d_18(
      pretrained=False,
      num_classes = 50,

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


#videos_root = os.path.join('/media/ysun1/Seagate_1/Dehao/data', 'UCF50_annotation')
videos_root = os.path.join('/media/ysun1/Seagate Expansion Drive/Dehao/data_dehao/UCF50_annotation', 'UCF50_annotation')

#videos_root = os.path.join('/media/ysun1/Seagate_1/Dehao/data', 'Sub_ucf50')
annotation_file = os.path.join(videos_root, 'annotations.txt')


preprocess = transforms.Compose([
        ImglistToTensor(),# list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(resize),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(resize),
        # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.455], std=[0.229, 0.224, 0.225]),
    ])

dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=num_seg,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )

sample = dataset[2]
frame_tensor = sample[0]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
label = sample[1]  # integer label

print('Video Tensor Size:', frame_tensor.size())

dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

#videos_root_val = os.path.join('/media/ysun1/Seagate_1/Dehao/data', 'UCF50_val_annotation')
#videos_root_val = os.path.join('/media/ysun1/Seagate_1/Dehao/data', 'Sub_val_ucf50')
videos_root_val = os.path.join('/media/ysun1/Seagate Expansion Drive/Dehao/data_dehao/UCF50_val_annotation', 'UCF50_val_annotation')
annotation_file_val = os.path.join(videos_root_val, 'annotations_val.txt')

dataset_val = VideoFrameDataset(
        root_path=videos_root_val,
        annotationfile_path=annotation_file_val,
        num_segments=num_seg,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )
val_dataloader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
#val_dataloader = dataloader


model = make_mvit()
#model = make_resnet3d_50()
#model = make_resnet3d_18()
#model = make_r2plus1d()
#model = make_r2plus1d_18()
#model = make_slowfast()
#model = make_x3d()

#model = make_csn()
#model = make_swin_transformer()
if torch.cuda.is_available():
    model = model.cuda()


optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epoch = 50
min_valid_loss = np.inf
for i in range(epoch):
    training_loss = 0.0
    model.train()
    for k, (video_batch, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            video_batch, labels = video_batch.cuda(), labels.cuda()

        video_batch = video_batch.permute(0,2,1,3,4) #exchange the dimension and the temporal dimension
        optimizer.zero_grad()
        pre = model(video_batch)
        loss = F.cross_entropy(pre, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    scheduler.step()

    valid_loss = 0.0
    total = 0.0
    correct = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for k, (video_batch, labels) in enumerate(val_dataloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            video_batch, labels = video_batch.cuda(), labels.cuda()
        video_batch = video_batch.permute(0, 2, 1, 3, 4)
        # Forward Pass
        pre = model(video_batch)
        # Find the Loss
        _, predicted = torch.max(pre.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = F.cross_entropy(pre, labels)
        # Calculate Loss
        valid_loss += loss.item()
        print(f'Accuracy of the network on the validation: {100 * correct / total} %')

    print(f'Epoch {i + 1} \t\t Training Loss: { training_loss / len(dataloader)} \t\t Validation Loss: { valid_loss / len(val_dataloader)}')
    print(f'Accuracy of the network on the validation: {100 * correct / total} %')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')




'''
slow_pathway = torch.index_select(
            video_batch.cpu(),
            2,
            torch.linspace(
                0, video_batch.cpu().shape[2] - 1, video_batch.cpu().shape[2] // 4
            ).long(),
        )

frame_list = [slow_pathway.cuda(), video_batch]
model(frame_list)
'''

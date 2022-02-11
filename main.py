import numpy as np
import torch
from torch import nn
from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pytorchvideo.models.vision_transformers
import torch.nn.functional as F
import torch.optim as optim

def make_kinetics_mvit():
  return pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(
      spatial_size=128,
      temporal_size=6,# RGB input from Kinetics
      depth=16, # For the tutorial let's just use a 50 layer network
      head_num_classes=50, # Kinetics has 400 classes so we need out final head to align
  )

videos_root = os.path.join('D:\\', 'UCF50_annotation')
#videos_root = os.path.join('D:\\', 'demo_dataset')
annotation_file = os.path.join(videos_root, 'annotations.txt')


preprocess = transforms.Compose([
        ImglistToTensor(),# list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(128),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(128),
        # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=6,
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
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

videos_root_val = os.path.join('D:\\', 'UCF50_val_annotation')
annotation_file_val = os.path.join(videos_root_val, 'annotations_val.txt')

dataset_val = VideoFrameDataset(
        root_path=videos_root_val,
        annotationfile_path=annotation_file_val,
        num_segments=6,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )
val_dataloader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
#val_dataloader = dataloader


model = make_kinetics_mvit()
if torch.cuda.is_available():
    model = model.cuda()
learning_rate = 0.001
weight_decay = 0.01
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
epoch = 10
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
          # print every 2000 mini-batches
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

    print(f'Epoch {i + 1} \t\t Training Loss: { training_loss / len(dataloader)} \t\t Validation Loss: { valid_loss / len(val_dataloader)}')
    print(f'Accuracy of the network on the validation: {100 * correct // total} %')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')









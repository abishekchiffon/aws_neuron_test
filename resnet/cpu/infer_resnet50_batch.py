import os
import time
import torch
import torch_neuron
import json
import numpy as np
from urllib import request
from torchvision import models, transforms, datasets
from time import time

batch_size = 5
## Create an image directory containing a small kitten
os.makedirs("./torch_neuron_test/images", exist_ok=True)
request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
 "./torch_neuron_test/images/kitten_small.jpg")

## Fetch labels to output the top classifications
request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
idx2label = []

with open("imagenet_class_index.json", "r") as read_file:
 class_idx = json.load(read_file)
 idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

## Import a sample image and normalize it into a tensor
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

eval_dataset = datasets.ImageFolder(
    os.path.dirname("./torch_neuron_test/"),
    transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])
)
image, _ = eval_dataset[0]
image = torch.tensor(image.numpy()[np.newaxis, ...])

## Load model
model_neuron = torch.jit.load( 'resnet50_neuron.pt' )

batch_image = image

for i in range((batch_size) - 1):
    batch_image = torch.cat( [batch_image, image], 0 )

## Since the first inference also load the model let's exclude it 
## from timing
results = model_neuron( batch_image )

## Predict for 100 loops
start = time()

loops = 100
for _ in range(loops):
    results = model_neuron( batch_image )
elapsed_time = time() - start
images_sec = loops * batch_image.size(0) / float(elapsed_time)

# Get the top 5 results
top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
top5_labels = [idx2label[idx] for idx in top5_idx]

print("Top 5 labels:\n {}".format(top5_labels) )
print("Completed {} operations in {} seconds for batches of size {} => {} images / second".format(loops * batch_image.size(0), round(elapsed_time,2), batch_size,round(images_sec,0) ) )

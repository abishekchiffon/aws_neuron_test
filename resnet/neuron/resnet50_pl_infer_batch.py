import os
from time import time
import torch
import torch_neuron
import json
import numpy as np
from urllib import request
from torchvision import models, transforms, datasets
from parallel import NeuronSimpleDataParallel

## Assuming you are working on and inf1.xlarge or inf1.2xlarge
num_neuron_cores = 4
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
model_neuron = NeuronSimpleDataParallel( '/home/ubuntu/infer_test/resnet50_neuron_b5.pt', num_neuron_cores, batch_size=batch_size  )

## Create a "batch" image with enough images to go on each of the four cores
batch_image = image

for i in range((num_neuron_cores * batch_size) - 1):
    batch_image = torch.cat( [batch_image, image], 0 )

## Since the first inference also loads the model to the chip let's exclude it 
## from timing
results = model_neuron( batch_image )

## Predict
start = time()
loops = 100
for _ in range(loops):
    results = model_neuron( batch_image )
elapsed_time = time() - start
images_sec = loops * batch_image.size(0) / elapsed_time

# Get the top 5 results
top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
top5_labels = [idx2label[idx] for idx in top5_idx]
print("Top 5 labels:\n {}".format(top5_labels) )
print("Completed {} operations in {} seconds => {} images / second".format( 
    loops * batch_image.size(0), round(elapsed_time, 2), round(images_sec,0) ) )

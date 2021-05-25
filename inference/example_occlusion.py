import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from captum.attr import Occlusion
from captum.attr import visualization as viz
model = torchvision.models.resnet18(pretrained=True).eval()

response = requests.get("https://image.freepik.com/free-photo/two-beautiful-puppies-cat-dog_58409-6024.jpg")
img = Image.open(BytesIO(response.content))
#img.show()
center_crop = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
])

normalize = transforms.Compose([
    transforms.ToTensor(),               # converts the image to a tensor with values between 0 and 1
    transforms.Normalize(                # normalize to follow 0-centered imagenet pixel rgb distribution
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
    )
])
input_img = normalize(center_crop(img)).unsqueeze(0)

occlusion = Occlusion(model)

strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
target=208                       # Labrador index in ImageNet
sliding_window_shapes=(3,45, 45)  # choose size enough to change object appearance
baselines = 0                     # values to occlude the image with. 0 corresponds to gray

attribution_dog = occlusion.attribute(input_img,
                                       strides = strides,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)
attribution_dog = np.transpose(attribution_dog.squeeze().cpu().detach().numpy(), (1,2,0))
for_heatmap = attribution_dog[:,:,2]
ax = sns.heatmap(for_heatmap)
plt.show()
#
# target=283,                       # Persian cat index in ImageNet
# attribution_cat = occlusion.attribute(center_crop(img),
#                                        strides = strides,
#                                        target=target,
#                                        sliding_window_shapes=sliding_window_shapes,
#                                        baselines=0)
# # Convert the compute attribution tensor into an image-like numpy array
# attribution_dog = np.transpose(attribution_dog.squeeze().cpu().detach().numpy(), (1,2,0))
#
# attribution_cat = np.transpose(attribution_cat.squeeze().cpu().detach().numpy(), (1,2,0))
#
# _ = viz.visualize_image_attr_multiple(attribution_cat,
#                                       center_crop(img),
#                                       ["heat_map", "original_image"],
#                                       ["all", "all"], # positive/negative attribution or all
#                                       ["attribution for cat", "image"],
#                                       show_colorbar = True
#                                      )
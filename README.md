Fine-tuned ResNet152 + YOLOv8 for brown bears detection.

# Task Description



# Solution Steps

## Data

### Data Processing

I used [**Roboflow**](https://roboflow.com/) service to manage data. Roboflow provides a streamlined workflow for data storage, labeling, augmentation, and formatting.

I used the service for the following:

- Data **storage and download API**;
- **Train-valid split** of the training set: 80% (290 images) and 20% (68 images);
- Data **augmentations** (increased the training set size by 3 times, up to 870 images):
    - Resize: fit in 896x896 (black edges)
    - Horizontal flip
    - Random crop: from 0% to 25%
    - Rotation: between -12° and +12°
    - Shear: ±5° horizontal, ±5° vertical



### Dataset Complement

After inferencing, I noticed several mistaken or low-confidence detections. I decided to **complement** the dataset by adding images with similar scenarios, e.g. "a brown bear hid behind the tree".

For image generation, I used a stable diffusion model ([link](https://stablediffusionweb.com/)). Probably, the same model was used to generate the competition dataset.

What's more probable is that the competition organizers have added noise to the data to change its distribution, so that tricks like supplementing wouldn't work well.

But seems like supplemnting helped me to correct the mistakes.

### Link to the Dataset

The final dataset version: [link](https://universe.roboflow.com/mydatasets-bqwxe/kaggle-kontur/dataset/8/images/?split=train)

No test set data is present in the dataset.
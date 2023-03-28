# Brown Bears Detection Model

*Help ResNet and YOLO find bears...*

<img src="https://user-images.githubusercontent.com/48735488/228231942-40697657-31ba-4d0b-bfda-37df918723f3.png" width=35% height=35%>

# Task Description

To train a model for brown bears detection at the images.

The [given dataset](https://www.kaggle.com/competitions/find-a-bear/data) consists of animals and nature images generated with a stable diffusion model.

# Solution Steps

## Data Processing

I used [**Roboflow**](https://roboflow.com/) service to manage data. Roboflow provides a streamlined workflow for data storage, labeling, augmentation, and formatting.

I used the service for the following:

- Data **storage and download API**.
- **Train-valid split** of the training set: 80% (290 images) and 20% (69 images).
- Data **augmentations** (increased the training set size by 3 times, up to 810 images):
    - Resize: fit in 896x896 (black edges)
    - Horizontal flip
    - Random crop: from 0% to 25%
    - Rotation: between -12° and +12°
    - Shear: ±5° horizontal, ±5° vertical
- Data **labeling** of newly generated data (increased the training set size by 20 * 3 = 60 images)

Original training sample             |  Augmented training sample
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/48735488/228058686-9c378bcf-3213-4c2c-9463-5d38beb176da.png" width=65% height=65%>  |  <img src="https://user-images.githubusercontent.com/48735488/228058764-d5b7f492-7b10-4d6c-8bf8-1923e1cabe02.png" width=65% height=65%>

## Dataset Complement

After inferencing, I noticed several mistaken and low-confidence detections. I decided to **complement** the dataset by adding images with similar scenarios, e.g. *"a brown bear hid behind the tree"*. In total, 20 * 3 = 60 new training samples were added to the dataset.

For image generation, I used [a stable diffusion model](https://stablediffusionweb.com/). Probably, the same model was used to collect the competition dataset. To the eye, generated samples are very similar to the dataset ones.

But that's not how the NNs work. I suppose the competition organizers could add noise to change the data distribution, so that tricks like dataset complement wouldn't work well.

Nevertheless, experiments shown that dataset complement helped to decrease the score (the less - the better).

Original test samples             |  Generated training samples
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/48735488/228027739-83cb4013-bb98-4ec3-818b-4b36180bb433.jpeg" width=65% height=65%>  |  <img src="https://user-images.githubusercontent.com/48735488/228030327-fe5c1826-2a39-46fc-bc2d-e79aed12575d.jpg" width=65% height=65%>
<img src="https://user-images.githubusercontent.com/48735488/228035453-c724cb99-1c49-4fbc-aae9-c2a35fd48a4f.jpeg" width=65% height=65%>  |  <img src="https://user-images.githubusercontent.com/48735488/228035434-8b256ee0-3af5-4a28-99bd-a30fa523c123.jpg" width=65% height=65%>

### Link to the Dataset

- The final dataset version: [link](https://universe.roboflow.com/mydatasets-bqwxe/kaggle-kontur/dataset/8/images/?split=train).

- No test set data is present in the dataset.

- I also corrected a couple of mistakes in initial training set labeling.

## Baseline

I chose **YOLOv8** model to detect bears. It's a SOTA model for object detection with simple interface.

As a baseline solution, I trained **YOLOv8** for 30 epochs on a raw dataset. The data had no augmentations, it was only resized to 320x320.

You can find the scores and metrics values for all the tested models in the end of this Readme.

## Binary Classification

I decided to help YOLO by "filtering out" the images without bears. For this purpose, I trained **a binary classifier with high recall value**.

I chose **ResNet152** as it's a strong CNN architecture and fine-tuned it to classify if there is a brown bear on an image, or not. The last *conv* block and a fully-connected head of ResNet152 were fine-tuned (in total, .. parameters out of ).

The unfreezed part of the ResNet152 is highlighted with red:
<img src="https://user-images.githubusercontent.com/48735488/228044361-05fc68c0-47c3-4fcf-b6db-96c1b5f7f72a.png" width=80% height=80%>

During fine-tuning, I focused on **Recall metric**, since it's important for a CNN to only perform a **preliminary filtering of images**.

CNN should "help" an object detection model by **discarding the images with no brown bears**. It shouldn't even have a goal of 100% accurate predictions, because such a good result may imply a risk of decreasing recall on unseen data.

I used the final dataset version (the link is above) for ResNet152 fine-tuning.

The code for fine-tuning is present in *resnet-fine-tuning.ipynb*.

## Object Detection

The best results were achieved with the following pipeline:
- Using the final dataset version (image size 896x896, complemented, augmented) for both ResNet152 and YOLOv8 training.
- "Filtering" the test set with the fine-tuned CNN.
- Inferencing the output with YOLOv8 trained for 120 epochs.

You can find the code for YOLOv8 training and inference in *brown-bears-detection.ipynb*.

# Results

Dataset and YOLOv8 metrics:

|YOLOv8 Model ID  |Image Size  |Augmented         |Complemented      |Train-valid Instances |Epochs  | Precision  | Recall    |   mAP50    | mAP50-95 |
|:---------------:|:----------:|:----------------:|:----------------:|:--------------------:|:------:|:-----------|:---------:|:-----------|:--------:|
|1                |   320x320  |:x:               |:x:               |     270-69           |30      |  0.797     |  0.9      |   0.938    |  0.901   |
|2                |   640x640  |:x:               |:x:               |     270-69           |30      |  0.804     |  0.9      |   0.921    |  0.895   |
|3                |   896x896  |:white_check_mark:|:x:               |     810-69           |30      |  0.663     |   0.984   |   0.879    |  0.47    |
|4                |   896x896  |:white_check_mark:|:white_check_mark:|     870-69           |30      |  0.911     |   0.955   |   0.976    |  0.682   |
|5                |   896x896  |:white_check_mark:|:white_check_mark:|     870-69           |70      |   0.956    | 1.0       |   0.993    |  0.62    |
|6                |   896x896  |:white_check_mark:|:white_check_mark:|     870-69           |120     |   0.875    | 1.0       |   0.982    |  0.691   |

Further training of YOLOv8 up to 250 epochs only deteriorated the results.

Model and competition score:

|YOLOv8 Model ID   |Binary Classification|Public Score |Private Score |Comments|
|:----------------:|:-------------------:|:-----------:|:------------:|:-------|
|1                 |   :x:               | 73.3        | 94.8         ||
|2                 |   :x:               | 71.0        | 95.0         ||
|3                 |   :x:               | 71.0        | 95.0         ||
|4                 |   :x:               | 87.9        | 57.1         ||
|5                 |   :x:               | 51.0        | 90.2         ||
|6                 |   :x:               | 44.4        | 72.0         ||
|2                 |   :white_check_mark:| 61.2        | 61.2         ||
|3                 |   :white_check_mark:| 61.2        | 61.2         ||
|4                 |   :white_check_mark:| 57.0        | 43.1         ||
|**5**             |   :white_check_mark:| **18.7**    | **19.1**     |**Private Top-2**|
|**6**             |   :white_check_mark:| **16.8**    | **46.7**     |**Public Top-1** |

### Observations

- Increasing the image size **from 320 to 640** gave some score decrease.
- Increasing the image size **from 640 to 896 and chosen augmentations** seem to have **no effect** on the score.
- **Binary classification** significantly decreases the score.
- **Dataset complement** helps decreasing the score.
- The model at the last row has such a high private score because ResNet classified a couple of dogs and polar bears images from the test set as positive ones. The CNN should aim at **Precision increase keeping the Recall as high as possible**.

# Space for improvement

There are ways to achieve even lower score:

- Complement the training set more, with both positive and negative samples.
- Search hyperparameters for ResNet152 training. 
- Try to find useful augmentations.

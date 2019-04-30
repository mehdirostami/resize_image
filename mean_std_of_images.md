

```python

```


```python
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image, ImageFilter, ImageStat
from torchvision import transforms, models, datasets
```

Import the file with labels and blurness and different stains.


```python
train_labels = pd.read_csv("/home/mr/SSC_case_study/train_label.csv")
```


```python
list(train_labels)
```




    ['image_name', 'count', 'blur', 'stain']




```python
main_path = "/home/mr/SSC_case_study/train1/"
```

Check what are the possible call counts in the data. 


```python
labels = np.unique(train_labels["count"])
```

Create new folders


```python
for c in labels:
    temp_path = main_path+"{}".format(c)
    os.mkdir(temp_path)
```

If you need to delete directories:


```python
# for folder in [x for x in np.arange(1, 101) if x not in labels.tolist()]:
#     os.rmdir(main_path+"{}".format(folder))
```

move files from the master folder to subfolders.


```python
# for name, c in zip(train_labels["image_name"], train_labels["count"]):
#     if name[0] == "A":
#         shutil.move(main_path+"{}".format(name), main_path+"{}/{}".format(c, name))
```

Create subfolders to put the images with different labels and other features in them to load them by data loader.


```python
for blur_level in ["A", "H", "P"]:
    os.mkdir("/home/mr/SSC_case_study/train2/{}/".format(blur_level))
```


```python
for blur_level in ["A", "H", "P"]:
    for stain in ["w1", "w2"]:
        os.mkdir("/home/mr/SSC_case_study/train2/{}/{}/".format(blur_level, stain))
```


```python
for blur_level in ["A", "H", "P"]:
    for stain in ["w1", "w2"]:
        for c in labels:
            os.mkdir("/home/mr/SSC_case_study/train2/{}/{}/{}".format(blur_level, stain, c))
```

Move original images to these subfolders:


```python
for blur_level in ["A", "H", "P"]:
    for stain in ["w1", "w2"]:
        for name, c, b, s in zip(train_labels["image_name"], train_labels["count"], train_labels["blur"], train_labels["stain"]):
            if name[0] == blur_level and stain[1] == str(s):
                shutil.move("/home/mr/SSC_case_study/train2/{}".format(name), "/home/mr/SSC_case_study/train2/{}/{}/{}/{}".format(blur_level, stain, c, name))
```

For machine learning, in particular CNNs, it is crucial to standardize the inputs. Since we have one channel images, we need to create a numeric matrix from each image and get the average and std of all pixels of each image, and taking mean of these means and stds. These will go into the transform function in pytorch.

First create dictionaries for each case that we want to analyze. Note that, we have 6 types of 


```python
mean1 = {}
std1 = {}

for name, c, b, s in zip(train_labels["image_name"], train_labels["count"], train_labels["blur"], train_labels["stain"]):
   
    mean1[blurs[b]+"_"+stains[s]] = []
    std1[blurs[b]+"_"+stains[s]] = []
```


```python
blurs = {1:"A", 23:"H", 48:"P"}
stains = {1:"w1", 2:"w2"}

for name, c, b, s in zip(train_labels["image_name"], train_labels["count"], train_labels["blur"], train_labels["stain"]):
    
    img = Image.open("/home/mr/SSC_case_study/train2/{}/{}/{}/{}".format(blurs[b], stains[s], c, name))
    pix = np.asarray(img) 

    mean1[blurs[b]+"_"+stains[s]] += [pix.mean()]
    std1[blurs[b]+"_"+stains[s]] += [pix.std()]

```


```python
means = {key: np.array(value).mean() for key, value in mean1.items()}
stds = {key: np.array(value).mean() for key, value in std1.items()}
```


```python
print(means)
print(stds)
```

    {'A_w2': 54.14309103531166, 'A_w1': 17.385638732316533, 'H_w1': 17.17658072225906, 'H_w2': 52.40316002155172, 'P_w1': 16.87478759118037, 'P_w2': 52.54237933106764}
    {'A_w2': 62.598789083719836, 'A_w1': 41.12060365027215, 'H_w1': 33.58713826838851, 'H_w2': 55.52143444400337, 'P_w1': 23.851820616433884, 'P_w2': 48.37113954672323}


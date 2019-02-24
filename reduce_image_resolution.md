
In this draft, the attempt is to get a folder of high-resolution images and convert them to low resolution images:

Let's pecify the size we want our low-resolution images to have:


```python
size = (64, 64)
```

Import the libraries:


```python
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
```

Let's go through all the pathes and names of the images in the train folder:


```python
path = Path("/home/mr/SSC_case_study/train/")

img_pths = [pth for pth in path.iterdir() if pth.suffix == '.TIF']
```

Now we want to resize each image. 

Better to save the new images in a new folder on the saving_path:


```python
saving_path = "/home/mr/SSC_case_study/low_resolution/train/"

for image in img_pths:
    temp_img = Image.open(image)
    temp_img = temp_img.resize(size, Image.ANTIALIAS)
    temp_img.save(saving_path + image.name, dpi=size)
```

Plot the last image for fun:


```python
plt.imshow(temp_img)
plt.show()
```


![png](output_10_0.png)


Here is the original image:


```python
plt.imshow(Image.open(image))
plt.show()
```


![png](output_12_0.png)


Repeat the whole procedure above for the test images. We need to make sure that size of images in the training and test steps is the same:


```python
path = Path("/home/mr/SSC_case_study/test/")

img_pths_test = [pth for pth in path.iterdir() if pth.suffix == '.TIF']
```


```python
saving_path = "/home/mr/SSC_case_study/low_resolution/test/"

for image in img_pths_test:
    temp_img_ = Image.open(image)
    temp_img_ = temp_img_.resize(size, Image.ANTIALIAS)
    temp_img_.save(saving_path + image.name, dpi=size)
```


```python
plt.imshow(temp_img_)
plt.show()
```


![png](output_16_0.png)



```python
plt.imshow(Image.open(image))
plt.show()
```


![png](output_17_0.png)


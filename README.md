# SAM_Web
This is the open source code for the final project of my Computer Vision and Pattern Recognition course at UESTC(2023 SUMMER). You can git clone it into your local environment, execute it, and feel the charm of SAM by interacting with it. Enjoy yourself, and have fun！

# some learning materials and resource are down below:

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

## A simple google colab example was available at [[`Here`](https://colab.research.google.com/github/Levi-Ackman/Sam_Web/Happiness/sam_interactive/colab.ipynb/)]

<p float="left">
  <img src="assets/Irving.jpg?raw=true" width="100%" />
  </p>
<p float="left">
  <img src="assets/Irving_mask.jpg?raw=true" width="100%" /> 
</p>
<p float="left">
  <img src="assets/kobe.jpg?raw=true" width="100%" />
</p>
<p float="left">
  <img src="assets/kobe_mask.jpg?raw=true" width="100%" /> 
</p>
<p float="left">
  <img src="assets/sea.jpg?raw=true" width="100%" />
</p>
<p float="left">
  <img src="assets/sea_mask.jpg?raw=true" width="100%" /> 
</p>

# Tutorial
## A simple, customized and optimized interactive segmentation web API based on the [Original Project Address:](https://github.com/facebookresearch/segment-anything.git/)

### A very basic flask app showing how to use segment-anything in browser.

### 1) First,Git clone this respository to your local laptop:

```
git clone https://github.com/Levi-Ackman/SAM_Web.git
```

### 2) Then,create the environment needed by : 
In conda:

```
conda env create -f environment.yml
```

Or In pip:

```
pip install -r requirements.txt
```

### 3) Download the model checkpoint:
Click the links below to download the checkpoint for the corresponding model type, and create a new folder checkpoint, put the checkpoint in ./checkpoint
(Personally recommand vit_h for better efect!!)

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

The easiest way of get the model checkpoint is by run in the terminal(Linux):

```
wget ... (url of the model adddress,such as:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth for vit_h)
```

### 4) Running the 'app.py' in different folder to experience different webui

Interactive segmentation (click,draw box,all item segmentation, erasion,and download your result):

```
cd sam_interactive
python app.py
```

Text prompt driven segmentaion (combine sam with clip--a powerful model generate text embedding aligned with image):

```
cd sam_clip
python app.py
```

### 5)
After run the command mentioned above, you may get a url like :

```
http://0.0.0.0:8244
```

for sam_interactive.

or

```
http://127.0.0.1:7861
```

and

```
https://42072a743a7935108c.gradio.live 
```

for sam_clip.

One thing notable here is that the link :https://42072a743a7935108c.gradio.live (with '.live' as suffix) can be accessed publicly, you can share your url with your friends!!
(A good opportunity to act tough).


# PS: A simplest way of experience the power of SAM 
After download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

first:

```

import cv2
your_image = cv2.imread('images/kobe.jpg')
your_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2RGB)
```

then:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```


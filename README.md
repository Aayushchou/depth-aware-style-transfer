# Depth Aware Style Transfer for Video

This code implements depth aware neural style transfer for application to videos. 

Completes locally on my Macbook Pro 2019 without gpu within 15 mins for 1 min video 🙆🏽‍♂️

## Installation

To install the package, please clone the repository and install via pip: 

```
git clone https://github.com/Aayushchou/depth-aware-style-transfer.git
cd image-styler
pip install -e .
```

## Procedure

The style transfer is done in the following steps: 

1. The video is split into frames using **split.py**. 
2. The frames are passed through the style transfer model with **transfer.py** to create stylized frames.
3. The directory containing the frames from step 1 must be passed to **transfer.py**.
4. Additionally, the style transfer has capability to take in two style images, one for the foreground and one for the background.
5. The max_width parameter determines the size of the images, 512 has worked best in my experience. 
6. There are some options to blur the background as well. 
7. This makes it easier to distinguish the main areas of focus in the video. 
8. The stylized frames are joined to a video using **join.py**. 

## Demo 

The video below demostrates one of the outputs from this: 

### Input Video

(Song is Mohe - 1 am)

https://user-images.githubusercontent.com/26253512/236692077-1f6cd9ce-3194-4165-9671-2f00b739892c.mp4

### Input Styles

<img src="https://user-images.githubusercontent.com/26253512/236692134-82004251-dbe6-46c5-a31a-cb6a2759b450.jpg" width="512" height="270">

<img src="https://user-images.githubusercontent.com/26253512/236692164-7f139c72-03ca-4f5f-b940-64296b92653e.jpg" width="512" height="270">


### Output Video

https://user-images.githubusercontent.com/26253512/236692306-6645895a-9860-4dc1-8053-20a77487cc14.mov

## Task Checklist

- [ ] add main.py to orchestrate end-to-end procedure 
- [ ] improvements to depth recognition
- [ ] try out more style image examples
- [ ] add front-end 


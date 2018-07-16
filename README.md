# Face_Images_GANs
Generate face images from GANs in keras

## Usage
- Download the wiki_crop or any other dataset of faces from here [Faces dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- Put all the images in "training_data" folder.
- Execute "resize_images.py" to resize them to 64x64x3 and saved to "blah/images/" automatically.
- Then run "keras_dcgan.py" to generate faces.

## Requirements
- Python3
- Tensorflow
- Keras
- OpenCV for image reading and writing

## Results
- Due to limited computing power I early stopped and got this as result but training on 8000 epochs will generate good quality samples
- ![blah](https://github.com/AsimMessi/Face_Images_GANs/blob/master/results/test1.png)

## License
- It is a free public tool to use

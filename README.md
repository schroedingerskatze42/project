# Attack face obfuscation with deep learning using convolution neural networks (CNN)

This repository has been created and is maintained by students of the _university of applied sciences Wedel, Germany_ ([Antonio Galeazzi](https://github.com/tonitassimo), [Till Hildebrandt](https://github.com/nobYsDarling)). contains different implementations of a CNN, that can be used to attack face obfuscation techniques.

## Frameworks used

### Keras
Keras is an open source neural network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, or MXNet. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible. It was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), and its primary author and maintainer is François Chollet, a Google engineer. ([Wikipedia](https://en.wikipedia.org/wiki/Keras))

### TensorFlow
TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks. It is used for both research and production at Google often replacing its closed-source predecessor, DistBelief. ([Wikipedia](https://de.wikipedia.org/wiki/TensorFlow))

## Techniques used

### Pixelization (mosaic technique)
![Original Image](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_mosaic_original.jpg "Original Image")
![Pixelization edge length 5px](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_mosaic_5.jpg "Pixelization edge length 5px")
![Pixelization edge length 10px](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_mosaic_10.jpg "Pixelization edge length 10px")


### Gaussian Blur
![Original Image](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_blur_original.jpg "Original Image")
![sigma = 3](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_blur_sigma_3.jpg "sigma = 3")
![sigma = 6](https://github.com/nobYsDarling/project/raw/master/documentation/Figures/introduction_blur_sigma_6.jpg "sigma = 6")


## Create test data and run
Test data is currently added to this repository. Under `databases`, subsets of the colorferet database as well as the facescrub database can be found there.

To recreate the test data from its `originals`, assert availability of ImageMagick in minimal Version 6 and locate the script `create_test_images.sh` within the database folder. The script will produce test data on bases of the folder `original`.

##### e.g.
```bash
cd databases/testimages_colorferet/
chmod +x create_images.sh
./create_images.sh
```

##### Synopsis
```bash
./create_images.sh [--no-grayscale] [--rgb] [--force]
```
* `--no-grayscale` do not create grayscale images
* `--rgb` do create rgb images
* `--force` override existing images

#### Implementations and execution
* `run_model_keras.py` trains and evaluates a cnn built with keras, colorferet database
* `run_model_srcnn.py` trains and evaluates a cnn built with tensorflow, colorferet database
* `run_model_tensorflow.py` trains and evaluates a cnn built with tensorflow, facescrub database

## Documentation
You find a `documentation.pdf` in the folder `documentation/`. You may recreate the `documentation.pdf` with LaTeX by
executing

```bash
./pdflatex documetation.tex
```

## Resources
 * [Keras](https://keras.io)
 * [TensorFlow](https://www.tensorflow.org)
 * [colorferet database](https://www.nist.gov/itl/iad/image-group/color-feret-database)
 * [facescrub database](http://vintage.winklerbros.net/facescrub.html)
 * [ImageMagick](https://www.imagemagick.org/script/index.php)
 * [LaTeX](https://www.latex-project.org/)

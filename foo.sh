#!/bin/bash

for f in $(find images -type f -iname "*.ppm");
do
    echo $f;

    d=$(echo $f | sed 's/images/images_50\/original/g');
    convert $f -scale 50% $d

#    d=$(echo $f | sed 's/images/images_50\/blurred_03/g');
#    convert $f -scale 50% -gaussian-blur 0x3 $d
    
#    d=$(echo $f | sed 's/images/images_50\/blurred_06/g');
#    convert $f -scale 50% -gaussian-blur 0x6 $d
    
#    d=$(echo $f | sed 's/images/images_50\/mosaic_05/g');
#    convert $f -scale 50% -scale 20% -scale 500% $d 
    
#    d=$(echo $f | sed 's/images/images_50\/mosaic_10/g');
#    convert $f -scale 50% -scale 10% -scale 1000% $d 


    d=$(echo $f | sed 's/images/images_50_grayscale\/original/g');
    convert $f -set colorspace Gray -separate -average -scale 50% $d
    
#    d=$(echo $f | sed 's/images/images_50_grayscale\/blurred_03/g');
#    convert $f -set colorspace Gray -separate -average -scale 50% -gaussian-blur 0x3 $d
    
#    d=$(echo $f | sed 's/images/images_50_grayscale\/blurred_06/g');
#    convert $f -set colorspace Gray -separate -average -scale 50% -gaussian-blur 0x6 $d
    
#    d=$(echo $f | sed 's/images/images_50_grayscale\/mosaic_05/g');
#    convert $f -set colorspace Gray -separate -average -scale 50% -scale 20% -scale 500% $d 
    
#    d=$(echo $f | sed 's/images/images_50_grayscale\/mosaic_10/g');
#    convert $f -set colorspace Gray -separate -average -scale 50% -scale 10% -scale 1000% $d 
done;

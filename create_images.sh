#!/bin/bash

PATH_ORIGINAL='./images'

PATH_DESTINATION_GRAYSCALE='./images_grayscale'
PATH_DESTINATION_30='./images_30'
PATH_DESTINATION_30_GRAYSCALE='./images_30_grayscale'
PATH_DESTINATION_50='./images_50'
PATH_DESTINATION_50_GRAYSCALE='./images_50_grayscale'

PATH_OPERATION_THUMBNAIL='/thumbnails'
PATH_OPERATION_BLUR_03='/blurred_03'
PATH_OPERATION_BLUR_06='/blurred_06'
PATH_OPERATION_BLUR_09='/blurred_09'
PATH_OPERATION_BLUR_12='/blurred_12'
PATH_OPERATION_BLUR_15='/blurred_15'
PATH_OPERATION_MOSAIC_05='/mosaic_05'
PATH_OPERATION_MOSAIC_10='/mosaic_10'
PATH_OPERATION_MOSAIC_15='/mosaic_15'
PATH_OPERATION_MOSAIC_20='/mosaic_20'
PATH_OPERATION_MOSAIC_25='/mosaic_25'

files=$(find ${PATH_ORIGINAL}${PATH_OPERATION_THUMBNAIL} -type f -iname *.ppm)
destinations="${PATH_ORIGINAL} \
              ${PATH_DESTINATION_GRAYSCALE} \
              ${PATH_DESTINATION_30} \
              ${PATH_DESTINATION_30_GRAYSCALE} \
              ${PATH_DESTINATION_50} \
              ${PATH_DESTINATION_50_GRAYSCALE}"
operations="${PATH_OPERATION_THUMBNAIL} \
            ${PATH_OPERATION_BLUR_03} \
            ${PATH_OPERATION_BLUR_06} \
            ${PATH_OPERATION_BLUR_09} \
            ${PATH_OPERATION_BLUR_12} \
            ${PATH_OPERATION_BLUR_15} \
            ${PATH_OPERATION_MOSAIC_05} \
            ${PATH_OPERATION_MOSAIC_10} \
            ${PATH_OPERATION_MOSAIC_15} \
            ${PATH_OPERATION_MOSAIC_20} \
            ${PATH_OPERATION_MOSAIC_25}"

function assert_dir {
    path=$1

    if [ ! -d ${path} ];
    then
        mkdir ${path}
    fi
}

function create_blurred_image {
    source=$1
    grayscale_part=$2
    scale_part=$3
    blur_radius=$4
    destination=$5

    convert ${source} \
            ${grayscale_part} \
            ${scale_part} \
            -gaussian-blur 0x${blur_radius} \
            ${destination}
}

function create_mosaic_image {
    source=$1
    grayscale_part=$2
    scale_part=$3
    mosaic_size=$4
    destination=$5

    convert ${source} \
            ${grayscale_part} \
            ${scale_part} \
            -scale $(( bc <<< "scale=100;100/${mosaic_size}" ))% \
            -scale $(( ${mosaic_size} * 100 ))% \
            ${destination}
}


function distribute_image_operation {
    file=${PATH_ORIGINAL}${PATH_OPERATION_THUMBNAIL}$1
    operation=$2
    grayscale_part=$3
    scale_part=$4
    destination=$5
    echo $file
    echo $destination
    exit 1
    case "${operation}" in
        ${PATH_OPERATION_BLUR_03})
            create_blurred_image ${file} ${grayscale} ${scale} 3 ${file_destination}
        ;;
        ${PATH_OPERATION_BLUR_06})
            create_blurred_image ${f} ${grayscale} ${scale} 6 ${file_destination}
        ;;
        ${PATH_OPERATION_BLUR_09})
            create_blurred_image ${f} ${grayscale} ${scale} 9 ${file_destination}
        ;;
        ${PATH_OPERATION_BLUR_12})
            create_blurred_image ${f} ${grayscale} ${scale} 12 ${file_destination}
        ;;
        ${PATH_OPERATION_BLUR_15})
            create_blurred_image ${f} ${grayscale} ${scale} 15 ${file_destination}
        ;;
        ${PATH_OPERATION_MOSAIC_05})
            create_mosaic_image ${f} ${grayscale} ${scale} 5 ${file_destination}
        ;;
        ${PATH_OPERATION_MOSAIC_10})
            create_mosaic_image ${f} ${grayscale} ${scale} 10 ${file_destination}
        ;;
        ${PATH_OPERATION_MOSAIC_15})
            create_mosaic_image ${f} ${grayscale} ${scale} 15 ${file_destination}
        ;;
        ${PATH_OPERATION_MOSAIC_20})
            create_mosaic_image ${f} ${grayscale} ${scale} 20 ${file_destination}
        ;;
        ${PATH_OPERATION_MOSAIC_25})
            create_mosaic_image ${f} ${grayscale} ${scale} 25 ${file_destination}
        ;;

        *)
            echo "some kind of error"
        exit 1
    esac
}

for d in ${destinations};
do
    # convert <img_in> -set colorspace Gray -separate -average <img_out>
    grayscale=''
    tmp=$d
    if [[ $tmp =~ ^.*_grayscale$ ]];
    then
        grayscale='-set colorspace Gray -separate -average'
    fi
    scale=''
    tmp=$d
    if [[ $tmp =~ ^.*_[0-9][0-9]*.*$ ]];
    then
        percentage=$(sed 's/^.*_\([0-9][0-9]*\).*$/\1/g' <<< $tmp)
        scale="-scale ${percentage}%"
    fi

    for o in ${operations};
    do
        if [ "${d}" != "${PATH_ORIGINAL}" ] || [ "${o}" != "${PATH_OPERATION_THUMBNAIL}" ];
        then
            echo "Start converting for ${d}${o}"

            for f in ${files};
            do
                filename=$(sed 's/^.*\(\/[^\/]*\)$/\1/g' <<< ${f})

                assert_dir ${d}${o}

                file_destination=${d}${o}${filename}

                distribute_image_operation ${filename} ${o} ${grayscale} ${scale} ${file_destination}
            done
        fi
    done
done

#    convert <img_in> -set colorspace Gray -separate -average <img_out>
#    echo ${f}

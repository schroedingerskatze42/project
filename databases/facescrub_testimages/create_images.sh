#!/bin/bash

# constants
IMAGES_PATH='./original'
DESTINATION_PATH_RGB='./rgb'
DESTINATION_PATH_GRAYSCALE='./grayscale'

FORCE=0
RGB=0
GRAYSCALE=1

SCALE=50

# argument parsing
ERROR=false
for i in "$@"
do
case ${i} in
    "--force")
        FORCE=1
        shift
    ;;
    "--rgb")
        RGB=1
        shift
    ;;
    "--no-grayscale")
        GRAYSCALE=0
        shift
    ;;
    *)
        echo "Unknown option"
        exit 1
    ;;
esac
done


# argument tests
if [ ${GRAYSCALE} -eq 0 ] && [ ${RGB} -eq 0 ]
then
    echo "Nothing to do."
    exit 1
fi

if [ ${FORCE} -eq 0 ] && ( ( [ ${GRAYSCALE} -eq 1 ] && [ -d ${DESTINATION_PATH_GRAYSCALE} ]) || ([ ${RGB} -eq 1 ] && [ -d ${DESTINATION_PATH_RGB} ] ) )
then
    echo "Destination path exists."
    exit 1
fi



# helper functions for image creation
get_filename_from_path () {
    sed 's/^.*\/\([^\/]*\)$/\1/g' <<< "$1"
}
destination_create_name() {
    sed "s|${IMAGES_PATH}|${DESTINATION_PATH_RGB}|g" <<< "$1"
}

destination_create_name_grayscale() {
    sed "s|${IMAGES_PATH}|${DESTINATION_PATH_GRAYSCALE}|g" <<< "$1"
}

destination_create_name_operation() {
    path=$1
    operation=$2
    parameter=$3

    echo "${path}/${operation}_${parameter}"
}


# assert needed directories
if [ ${RGB} -eq 1 ];
then
    original_dir="${DESTINATION_PATH_RGB}/original"
    echo "assert directory ${original_dir}"
    mkdir -p ${original_dir}

    blurred_03_dir=$(destination_create_name_operation ${DESTINATION_PATH_RGB} 'blur' '03' )
    echo "assert directory ${blurred_03_dir}"
    mkdir -p ${blurred_03_dir}

    mosaic_05_dir=$(destination_create_name_operation ${DESTINATION_PATH_RGB} 'mosaic' '05' )
    echo "assert directory ${mosaic_05_dir}"
    mkdir -p ${mosaic_05_dir}
fi
if [ ${GRAYSCALE} -eq 1 ];
then
    original_dir="${DESTINATION_PATH_GRAYSCALE}/original"
    echo "assert directory ${original_dir}"
    mkdir -p ${original_dir}

    blurred_03_dir=$(destination_create_name_operation ${DESTINATION_PATH_GRAYSCALE} 'blur' '03' )
    echo "assert directory ${blurred_03_dir}"
    mkdir -p ${blurred_03_dir}

    mosaic_05_dir=$(destination_create_name_operation ${DESTINATION_PATH_GRAYSCALE} 'mosaic' '05' )
    echo "assert directory ${mosaic_05_dir}"
    mkdir -p ${mosaic_05_dir}
fi

num_of_files=$(find "${IMAGES_PATH}" -type f -iname "*.ppm" | wc -l)
i=0
for f in $(find "${IMAGES_PATH}" -type f -iname "*.ppm");
do
    i=$(( $i + 1 ));

    if [ $(( $i % 10 )) -eq 0 ] || [ ${i} -eq ${num_of_files} ];
    then
        echo "Converting images ${i}/${num_of_files}..."
    fi

    if [ ${RGB} -eq 1 ];
    then
        filename=$(get_filename_from_path ${f})

#        path="${DESTINATION_PATH_RGB}/original"
#        d="${path}/${filename}"
#        convert ${f} -adaptive-resize ${SCALE}% ${d}

        path=$(destination_create_name_operation ${DESTINATION_PATH_RGB} 'blur' '03')
        d="${path}/${filename}"
        convert ${f} -adaptive-resize ${SCALE}% -gaussian-blur 0x3 ${d}

        path=$(destination_create_name_operation ${DESTINATION_PATH_RGB} 'mosaic' '05')
        d="${path}/${filename}"
        convert ${f} -adaptive-resize ${SCALE}% -scale 20% -scale 500% -resize "64x96^" -gravity center -crop 64x96+0+0 ${d}
    fi

    if [ ${GRAYSCALE} -eq 1 ];
    then
        ppm_name=$(get_filename_from_path ${f})
        pgm_name=$(sed 's/.ppm$/.pgm$/g' <<< "$ppm_name")  # ImageMagick is sensitive to file suffix

#        path="${DESTINATION_PATH_GRAYSCALE}/original"
#        d="${path}/${pgm_name}"
#        convert ${f} -set colorspace Gray -separate -average -adaptive-resize ${SCALE}% ${d}
#        mv ${d} ${path}/${ppm_name}

        path=$(destination_create_name_operation ${DESTINATION_PATH_GRAYSCALE} 'blur' '03')
        d="${path}/${pgm_name}"
        convert ${f} -set colorspace Gray -separate -average -adaptive-resize ${SCALE}% -gaussian-blur 0x3 ${d}
        mv ${d} ${path}/${ppm_name}

        path=$(destination_create_name_operation ${DESTINATION_PATH_GRAYSCALE} 'mosaic' '05')
        d="${path}/${pgm_name}"
        convert ${f} -set colorspace Gray -separate -average -adaptive-resize ${SCALE}% -scale 20% -scale 500% -resize "64x96^" -gravity center -crop 64x96+0+0 ${d}
        mv ${d} ${path}/${ppm_name}
    fi
done;

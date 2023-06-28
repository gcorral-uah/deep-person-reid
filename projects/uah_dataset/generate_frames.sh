#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# set -x

usage() {
    echo "$0 usage:"
    echo "Generate frames pictures in FRAMES/"
    echo "By default it only generate the frames where there is not a video without an adjuncent FRAMES folder"
    echo "Use -d to delete the directories of previous frames."
    echo "Use -r to recreate the directories of previous frames."
    echo "Use -x to delete the directories of previous xml."
    exit 0
}

delete="no"
recreate="no"
delete_xml="no"
while getopts ":drh" flag; do
    case "${flag}" in
        d) delete="yes";;
        r) recreate="yes";;
        x) delete_xml="yes";;
        h)
            usage
            exit 0;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

echo "Delete is $delete"
echo "Recreate is $recreate"
echo "Delete xml is $delete_xml"

if [[ $recreate =~ "yes" && $delete =~ "yes" ]]; then
    echo "Can't use recreate and delete at the same time"
    exit
fi

if [[ $delete_xml =~ "yes" ]]; then
    echo "Deleting xml. This is an exclusive operation."
    exit
fi


for DIR in */; do
    if [[ "$DIR" =~ "other" ]]; then
        # Ignore the other folder, it doesn't have any videos
        continue
    fi

    echo "Entering $DIR"

    pushd $DIR

    if [[ $delete_xml =~ "yes" ]]; then
        echo "Going to remove ${DIR}xml"
        rm -rf xml/
        # We have to popd here becuase we want to return to the previous folder, even if the xml folder exists.
        popd
        continue
    fi

    if [[ $delete =~ "yes" ]]; then
        echo "Going to remove ${DIR}FRAMES"
        rm -rf FRAMES/
        # We have to popd here becuase we want to return to the previous folder, even if the FRAMES folder exists.
        popd
        continue
    fi

    if [[ $recreate =~ "yes" ]]; then
        echo "Going to remove ${DIR}FRAMES to recreate the FRAMES"
        rm -rf FRAMES/
    fi

    if [[ -d "FRAMES" ]]; then
        echo "Skipping $DIR because it has a FRAMES folder and you didn't specify -r"
        # We have to popd here becuase we want to return to the previous folder, even if the FRAMES folder exists.
        popd
        continue
    fi

    # Use an array to hold the glob, it's probably the best way to do it...
    video_file=( *.MP4 )
    echo "Globbed the video file $video_file in $DIR"

    if (( ${#video_file[@]} == 0 )); then
        # If there has been an error and the array is empty continue
        echo "There is an empty dir $DIR"
        continue
    fi

    # The array only has one element, so access it directly.
    ffmpeg -i "${video_file[0]}" "%06d.jpg"
    mkdir FRAMES

    # This doen't need to be quoted, becaused we have generated the names without spaces or rare characters.
    mv *.jpg FRAMES/

    popd
done

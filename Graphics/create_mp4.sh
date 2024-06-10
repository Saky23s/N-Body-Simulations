#!/bin/bash 

#Delete old data
if [ -d img ]; then
  rm -f img/*.png
else
  mkdir img
fi

cargo run
n= ls -1q img/*.png | wc -l

ffmpeg -f image2 -framerate 25 -i img/%05d.png -vcodec libx264 -crf 22 simulation.mp4
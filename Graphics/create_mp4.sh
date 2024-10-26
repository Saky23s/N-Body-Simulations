#!/bin/bash 

#Delete old data
if [ -d img ]; then
  rm -f img/*.png
else
  mkdir img
fi

cargo run

if [ -f simulation.mp4 ]; then
  rm -f simulation.mp4
else
  mkdir img
fi

n= ls -1q img/*.png | wc -l

ffmpeg -f image2 -framerate 60 -i img/%05d.png -vf "scale=1920:1080" -vcodec libx264 -pix_fmt yuv420p -crf 28 -movflags faststart -preset medium simulation.mp4

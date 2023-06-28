The command to extract the frames is
```bash
ffmpeg -i video.mp4 %06d.jpg
```

The gt2xml.m script runs on octave, so you don't need MATLAB installed.

You can copy the script generate_frames.sh to the dataset folder and use to extract the frames.

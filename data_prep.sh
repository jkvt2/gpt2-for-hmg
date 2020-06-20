cd data
mkdir raw
ffmpeg -i videoplayback.webm -vsync 0 raw/out%d.png
python sort_frames.py
cd ../hmr
python extract.py

OPENPOSE=/path/to/openpose

cd data
mkdir raw json hmr
ffmpeg -i videoplayback.webm -vsync 0 raw/out%d.png
python sort_frames.py

cwd=$(pwd)
cd $OPENPOSE
for file in $cwd/preproc/*; do
  build/examples/openpose/openpose.bin --image_dir $cwd/preproc/"${file##*/}"/ --write_json $cwd/json/"${file##*/}"/
done

cd ../hmr
python extract.py

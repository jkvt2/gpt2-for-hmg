# GPT2 for Human Motion Generation
Using GPT2 to build a human motion model!

Kanazawa et al.'s End-to-end Recovery of Human Shape and Pose allows us to make pseudo ground truth labels of human shape and pose sequences: \
![Example of generic walk sequence](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/hmr/00_generic.gif) \
We can train GPT2 on such sequences! Trained using (90 of the) the 100 Ways to Walk youtube video, we can generate plausible motions on the test set (the remaining 10 ways to walk). NOTE: that the ground truth ends first as the generated sequences are longer. \
![Inferred scuba diver](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/nice/32_scuba_diver.gif) \
(The minister for silly walks takes up) Scuba diving\
![Inferred moonwalk](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/nice/46_moonwalk.gif) \
Moonwalk\
![Inferred windstorm](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/nice/74_in_a_wind_storm.gif) \
In a wind storm

Although these examples may not match the ground truth, they still look quite plausible. Sometimes, however, the entire point is missed:\
![Inferred royal guard](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/fail/49_royal_guard.gif) \
Royal guard has given up marching and just wants to stroll\
![Inferred stepped in something](https://github.com/jkvt2/gpt2-for-hmg/blob/master/figures/fail/93_stepped_in_something.gif) \
Stepped-in-something guy knows that his feet are sticky but he wants to maintain his "tough" image

### Requirements
Modified hmr and GPT2 folders are included in here, but please follow their reqs/installation instructions
- [hmr](https://github.com/akanazawa/hmr) (including downloading the pre-trained models)
- [GPT2](https://github.com/ConnorJL/GPT2)

Separately, you'll need an installation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for hmr.

### Steps
1. Download the [100 Ways to Walk](https://www.youtube.com/watch?v=HEoUhlesN9E&t=4s) video. This is "videoplayback.webm" in data_prep.sh:
```
ffmpeg -i videoplayback.webm -vsync 0 raw/out%d.png
```
2. Modify path to openpose in data_prep.sh
3. Run data_prep.sh. This should give you the following folders in data/:
- preproc: this is just the different walks in .png form, sorted into folders
- json: this is the output from openpose, which hmr relies upon. Gives you the bounding box of the person.
- hmr: this is the pickled extracted 85-dimensional representation of the person, using hmr.
4. Run train_gpt2.sh. This will give you the checkpoints and logs folders in GPT2/.
5. Run infer_gpt2.sh. This will give you:
- GPT2/logs/train and GPT2/logs/test: these are the predicted representations for each frame, for 10 examples from the test and training sets. Note: the first n_ctx (default:30, set in GPT2/small.json) frames are the given context and are not predicted.
- data/gpt_vis/train and data/gpt_vis/test. You should find 101 images for each walk as we generated 90 frames (n_pred) but start visualisation from frame 19 (to save some runtime).

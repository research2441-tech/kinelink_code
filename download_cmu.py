"""Download the 102 CMU motion-capture clips (BVH conversion) used in the study into ./bvh."""
import os, urllib.request
import gen_data as G

BASE = 'https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data'
os.makedirs('bvh', exist_ok=True)
clips = sorted(set(G.TRAIN + G.VAL + G.TEST_WALK + G.TEST_CARE + G.TEST_RUN))
for c in clips:
    fn = f'bvh/{c}.bvh'
    if not os.path.exists(fn):
        urllib.request.urlretrieve(f'{BASE}/0{c.split("_")[0]}/{c}.bvh', fn)
print(len(clips), 'clips available in ./bvh')

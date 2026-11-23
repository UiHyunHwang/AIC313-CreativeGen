"""Preview the outward camera path using the archived three-board cycle."""
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent
source=root.parent/"hero_cycle_varied/ironman_cycle.gif"
with Image.open(source) as im:
    palette=im.copy();frames=[];ms=0;holds=[];ends=[]
    for i in range(im.n_frames):
        im.seek(i);duration=im.info["duration"]
        if duration>=1000: holds.append(i)
        frames.append(im.convert("RGB").quantize(palette=palette,dither=Image.Dither.NONE))
        ms+=duration
    assert ms==30020 and len(holds)==3 and holds[0]==0
    assert frames[-1].tobytes()==frames[0].tobytes()
    ends=holds+[len(frames)-1]
play=[];times=[]
for leg in (0,2,1):
    play.append(frames[ends[leg+1]]);times.append(5500)
    movement=frames[ends[leg]+1:ends[leg+1]][::-1];play.extend(movement)
    times.extend((10*np.diff(np.rint(np.linspace(0,450,len(movement)+1)).astype(int))).tolist())
play.append(play[0]);times.append(20)
gif=root/"outward_preview.gif"
play[0].save(gif,save_all=True,append_images=play[1:],loop=0,disposal=1,optimize=False,duration=times)
with Image.open(gif) as check:
    stops=[];ms=0
    for j in range(check.n_frames):
        check.seek(j);d=check.info["duration"]
        if d>=1000:stops.append(check.convert("RGB").tobytes())
        ms+=d
    assert stops==[frames[holds[i]].convert("RGB").tobytes() for i in (1,0,2)]
    assert check.convert("RGB").tobytes()==stops[0] and ms==30020

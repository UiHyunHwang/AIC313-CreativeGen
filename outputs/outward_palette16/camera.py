"""Outward camera from the project outward_hero_cycle renderer, parameterized for a 16x16 field."""
from pathlib import Path
import numpy as np
from PIL import Image
from sdxl.hero_cycle_render import LANCZOS,SIDE,project,digest

ROOT=Path(__file__).resolve().parent
GRID=16
HEROES=("spider_man","iron_man","captain_america")
def xy(i,side):
    return (i%GRID*side,i//GRID*side)

def field_width(t):
    ease=t*t*t*(10-15*t+6*t*t)
    return GRID**(ease-1)

def project_layers(levels,slots,outer,bounds):
    image=Image.new("RGB",(SIDE,SIDE))
    x,y,side=0.,0.,1.
    for _ in range(4):
        project(image,levels[outer],bounds,x,y,side)
        side/=GRID
        dx,dy=xy(slots[outer],side)
        x+=dx;y+=dy
        outer=(outer-1)%len(levels)
    return image

def film(levels,slots,*,heroes=HEROES,output_dir=None,filename="spider_iron_captain_zoom_out.gif",palette=None,palette_at=None):
    count_stages=len(heroes)
    assert count_stages==len(levels)==len(slots)
    output_dir=ROOT if output_dir is None else output_dir
    # The camera width in parent coordinates increases from one cell to the full parent.
    widths=[field_width(k/91) for k in range(92)]
    assert widths[0]==1/GRID and widths[-1]==1 and all(a<b for a,b in zip(widths,widths[1:]))
    holds=[project_layers(levels,slots,j,(0.,0.,1.)) for j in range(count_stages)]
    def frame(inner,t):
        outer=(inner+1)%count_stages
        if t==0:return holds[inner]
        if t==1:return holds[outer]
        width=field_width(t)
        x,y=xy(slots[outer],1/GRID)
        mix=(1-width)/(1-1/GRID)
        return project_layers(levels,slots,outer,(x*mix,y*mix,width))
    if palette is None:
        atlas=Image.new("RGB",(8*100,3*count_stages*100))
        for j in range(count_stages):
            for k in range(24):
                idx=j*24+k
                atlas.paste(frame(j,k/23).resize((100,100),LANCZOS),(idx%8*100,idx//8*100))
        palette=atlas.quantize(256,method=Image.Quantize.MEDIANCUT)
    def quantize(im,j,t):
        active=palette if palette_at is None else palette_at(j,t)
        return im.quantize(palette=active,dither=Image.Dither.FLOYDSTEINBERG)
    stops=[quantize(im,j,0.) for j,im in enumerate(holds)];frames=[];times=[]
    for j in range(count_stages):
        next_j=(j+1)%count_stages;frames.append(stops[j]);times.append(5500)
        movement=[quantize(frame(j,k/91),j,k/91) for k in range(1,91)]
        while movement and movement[0].tobytes()==stops[j].tobytes():movement.pop(0)
        while movement and movement[-1].tobytes()==stops[next_j].tobytes():movement.pop()
        assert movement
        frames.extend(movement)
        times.extend((10*np.diff(np.rint(np.linspace(0,450,len(movement)+1)).astype(int))).tolist())
    frames.append(quantize(frame(count_stages-1,1),count_stages-1,1.));times.append(20)
    path=output_dir/filename
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=times,loop=0,optimize=False,disposal=1)
    elapsed=0;observed=[];timings=[]
    with Image.open(path) as im:
        assert im.info["loop"]==0
        for k in range(im.n_frames):
            im.seek(k);dt=im.info["duration"]
            if dt>=1000:
                observed.append(im.convert("RGB").tobytes());timings.append((elapsed,dt))
            elapsed+=dt
        assert im.convert("RGB").tobytes()==observed[0]
        count=im.n_frames
    assert observed==[s.convert("RGB").tobytes() for s in stops]
    assert timings==[(10000*j,5500) for j in range(count_stages)] and elapsed==10000*count_stages+20
    # Save actual camera samples: beginning, halfway out, and wide for each reveal.
    strip=Image.new("RGB",(3*240,count_stages*240))
    for j in range(count_stages):
        for k,t in enumerate((0,.5,1)):
            strip.paste(frame(j,t).resize((240,240),LANCZOS),(k*240,j*240))
    strip.save(output_dir/"zoom_sequence.jpg",quality=93)
    return dict(order=[*heroes,heroes[0]],camera="outward",width_per_leg=[1/GRID,1.0],
        frames=count,duration_ms=elapsed,holds=timings,first_last_identical=True,gif_sha256=digest(path))


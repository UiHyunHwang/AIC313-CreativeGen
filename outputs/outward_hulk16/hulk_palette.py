"""Keep the old stop palettes and give Hulk proper green tones in a GIF."""
from functools import lru_cache
import numpy as np
from PIL import Image
from outputs.outward_palette16.camera import field_width

def indexed(table):
    im=Image.new("P",(1,1))
    im.putpalette(np.asarray(table,dtype=np.uint8).ravel().tolist())
    return im

def scene_palettes(approved,views):
    base=np.array(approved.getpalette(),dtype=np.float64).reshape(256,3)
    usage=np.ones(256)
    for view in views[1:4]:
        pixels=np.asarray(view.quantize(palette=approved))
        usage+=np.bincount(pixels.ravel(),minlength=256)
    # Remove the least costly redundant entries. Recompute distances after each removal.
    squared=np.sum((base[:,None]-base[None,:])**2,axis=2)
    np.fill_diagonal(squared,np.inf)
    freed=[]
    for _ in range(32):
        cost=usage*squared.min(axis=1)
        if freed:cost[freed]=np.inf
        idx=int(cost.argmin());freed.append(idx);squared[:,idx]=np.inf
    samples=[]
    for j in (0,4):
        rgb=np.asarray(views[j]).astype(np.int16)
        r,g,b=np.moveaxis(rgb,-1,0)
        green=(g>=r-5)&(g-b>12)
        samples.append(rgb[green][::3])
    points=np.concatenate(samples).astype(np.uint8)
    if len(points)<32:raise ValueError("Insufficient green samples to fit the new palette")
    # Retain the measured hue distribution from the actual diffusion outputs.
    strip=Image.fromarray(points.reshape(1,-1,3))
    entries=np.asarray(strip.quantize(32,method=Image.Quantize.MEDIANCUT).getpalette()).reshape(-1,3)[:32]
    extended=base.copy()
    unused=list(freed)
    # Assign each new entry to the nearest available old one for smooth interpolation.
    for color in entries:
        closest=min(unused,key=lambda k:np.sum((base[k]-color)**2))
        extended[closest]=color;unused.remove(closest)
    @lru_cache(maxsize=33)
    def mix(amount):
        if amount==0:return approved
        return indexed(np.rint(base+(extended-base)*(amount/32)))
    def palette_at(inner,t):
        outer=(inner+1)%len(views)
        start=int(inner in (0,4));end=int(outer in (0,4))
        if start==end:return mix(32*start)
        # The outgoing board occupies this fraction of the animation frame.
        area=(1/(16*field_width(t)))**2
        old_share=np.clip((area-1/256)/(1-1/256),0,1)
        return mix(round(32*(start*old_share+end*(1-old_share))))
    return palette_at,dict(replaced_indices=freed,added_rgb=entries.tolist(),sampled_green_pixels=len(points))

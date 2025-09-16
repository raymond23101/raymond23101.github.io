import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
import os

# ====== hyperparameters =========
DATA_DIR  = 'data'
OUT_DIR   = 'out'
METRIC    = 'ncc'
RADIUS    = 15
CROP_FRAC = 0.20
PYRAMID   = False
MIN_SIDE  = 64
TRIAL_NUM = 2
# ================================

def split_bgr(im):
    h = np.floor(im.shape[0] / 3.0).astype(int)
    return im[:h], im[h:2*h], im[2*h:3*h]

def internal_crop(im, frac): # Used final advice of computing only on internal pixels
    if frac <= 0: return im
    H, W = im.shape
    ch, cw = int(H*frac), int(W*frac)
    return im[ch:H-ch, cw:W-cw]

def overlap_views(ref, mov, dy, dx):
    H, W = ref.shape
    # Calculate starting positions
    ref_start_y = max(0, dy)
    ref_start_x = max(0, dx)
    mov_start_y = max(0, -dy)
    mov_start_x = max(0, -dx)
    # Calculate overlap
    overlap_h = min(H - ref_start_y, H - mov_start_y)
    overlap_w = min(W - ref_start_x, W - mov_start_x)
    # Check if overlap exists
    if overlap_h <= 0 or overlap_w <= 0:
        return None, None

    ref_region = ref[ref_start_y:ref_start_y+overlap_h, ref_start_x:ref_start_x+overlap_w]
    mov_region = mov[mov_start_y:mov_start_y+overlap_h, mov_start_x:mov_start_x+overlap_w]
    return ref_region, mov_region

def L2(a, b):
    d = a - b
    return float(np.mean(d*d))

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    a_flat = a.flatten();b_flat = b.flatten()
    return -float(np.dot(a_flat/np.linalg.norm(a_flat), b_flat/np.linalg.norm(b_flat)))

def align_single_scale(ref, mov, radius, metric, crop_frac):
    scorer = ncc if metric.lower() == 'ncc' else L2
    current = (0, 0, np.inf)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            rV, mV = overlap_views(ref, mov, dy, dx)
            if rV is None: continue
            rC = internal_crop(rV, crop_frac); mC = internal_crop(mV, crop_frac)
            if rC.size == 0: continue
            s = scorer(rC, mC)
            if s < current[2]: current = (dy, dx, s)
    return int(current[0]), int(current[1])

def align_pyramid(ref, mov, base_radius, metric, crop_frac, min_side):
    dy, dx = 0, 0
    # build pyramid (smallest last)
    pyr_r = [ref]; pyr_m = [mov]
    while min(pyr_r[-1].shape) > min_side:
        pyr_r.append(rescale(pyr_r[-1], 0.5, anti_aliasing=True))
        pyr_m.append(rescale(pyr_m[-1], 0.5, anti_aliasing=True))
        #if len(pyr_r) >= 6: break  # cap levels

    for lvl in reversed(range(len(pyr_r))):
        R, M = pyr_r[lvl], pyr_m[lvl]
        if lvl != len(pyr_r) - 1: dy *= 2; dx *= 2
        current = (dy, dx, np.inf)
        scorer = ncc if metric.lower() == 'ncc' else L2
        for ddy in range(dy-base_radius, dy+base_radius+1):
            for ddx in range(dx-base_radius, dx+base_radius+1):
                rV, mV = overlap_views(R, M, ddy, ddx)
                if rV is None: continue
                rC = internal_crop(rV, crop_frac); mC = internal_crop(mV, crop_frac)
                if rC.size == 0: continue
                s = scorer(rC, mC)
                if s < current[2]: current = (ddy, ddx, s)
        dy, dx, _ = current
    return int(dy), int(dx)

def roll_shift(im, dy, dx):
    return np.roll(np.roll(im, dy, axis=0), dx, axis=1)

# --- main ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', DATA_DIR)
out_path = os.path.join(script_dir, '..', OUT_DIR, f'trial_{TRIAL_NUM}')
os.makedirs(out_path, exist_ok=True)
image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.tif'))]
image_files.sort()

print(f"Found {len(image_files)} image files to process:")
for imname in image_files:
    print(f"\nProcessing {imname}...")
    im_path = os.path.join(data_path, imname)
    im = skio.imread(im_path)
    im = sk.img_as_float(im)
    b, g, r = split_bgr(im)
    
    is_tif = imname.lower().endswith(('.tif'))
    use_pyramid = PYRAMID or is_tif
    
    if use_pyramid:
        reason = "TIF file" if is_tif else "PYRAMID=True"
        print(f"  Using pyramid alignment ({reason}: {im.shape}, radius: {RADIUS})")
        dyg, dxg = align_pyramid(b, g, RADIUS, METRIC, CROP_FRAC, MIN_SIDE)
        dyr, dxr = align_pyramid(b, r, RADIUS, METRIC, CROP_FRAC, MIN_SIDE)
    else:
        print(f"  Using single-scale alignment (JPG file: {im.shape}, radius: {RADIUS})")
        dyg, dxg = align_single_scale(b, g, RADIUS, METRIC, CROP_FRAC)
        dyr, dxr = align_single_scale(b, r, RADIUS, METRIC, CROP_FRAC)
    
    ag = roll_shift(g, dyg, dxg)
    ar = roll_shift(r, dyr, dxr)
    color = np.clip(np.dstack([ar, ag, b]), 0, 1)
    
    base_name = os.path.splitext(imname)[0]
    output_filename = f"{base_name}_colorized.jpg"
    output_path = os.path.join(out_path, output_filename)
    
    color_uint8 = (color * 255).astype(np.uint8)
    skio.imsave(output_path, color_uint8)
    
    print(f"  G aligned to B: (x, y) = ({dxg}, {dyg})")
    print(f"  R aligned to B: (x, y) = ({dxr}, {dyr})")
    print(f"  Saved: {output_filename}")
print(f"\nAll {len(image_files)} images processed successfully!")

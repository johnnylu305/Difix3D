import os
import numpy as np
from PIL import Image, ImageOps
from model import Difix
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import random, math


# Paths
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp2" 
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_29999"
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/old_mask_not_undist/patio_high_first/renders/render_29999"
input_dir = "/home/johnny305/Documents/dfpaint/dataset/results_org/patio_high_first/renders/render_29999"
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp3" 
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_difixsls2_noreference_29999"
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/old_mask_not_undist/patio_high_first/renders/render_difixsls_ref_29999"
output_dir = "/home/johnny305/Documents/dfpaint/dataset/results_org/patio_high_first/renders/difixsls_bs2_2gpu_pretrain_mulref1_src"

# --- INPUTS (edit these) ---
CKPT = "/home/johnny305/Documents/dfpaint/difix_sls/bs2_2gpu_pretrain_mulref1_src/model_25001.pkl"
LORA_RANK_VAE = 4          # --lora_rank_vae (not directly used here but keep for reference)
TIMESTEP = 199             # --timestep
MV_UNET = True            # set True if you trained with --mv_unet
NV = 1

os.makedirs(output_dir, exist_ok=True)

# Build your trained model wrapper
model = Difix(
    pretrained_name=None,
    pretrained_path=CKPT,
    timestep=TIMESTEP,
    mv_unet=MV_UNET,
)
model.set_eval()


# Original difix
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")


def load_pipe_weights_into_model(pipe, model, report: bool = True):
    """
    Load weights from a DifixPipeline (pipe) into a Difix model (model).

    Args:
        pipe: DifixPipeline instance (HuggingFace Diffusers style).
        model: Difix model instance (your training version).
        report: If True, print a summary of loaded/missing/unexpected keys.
    """

    results = {}

    def load_component(dst_module, src_module, name):
        dst_sd = dst_module.state_dict()
        src_sd = src_module.state_dict()
        missing, unexpected = dst_module.load_state_dict(src_sd, strict=False)
        results[name] = {"missing": missing, "unexpected": unexpected}

        if report:
            if not missing:
                status = "✅ OK"
            else:
                status = "❌ NOT OK"
            print(f"{name}: {dst_module.__class__.__name__} "
                  f"(params: {len(dst_sd)}) --> {status}")
            if missing:
                print(f"   Missing ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"   Unexpected ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    # UNet
    load_component(model.unet, pipe.unet, "unet")

    # VAE
    load_component(model.vae, pipe.vae, "vae")

    # Text encoder
    load_component(model.text_encoder, pipe.text_encoder, "text_encoder")

    # Copy over tokenizer / scheduler (not modules)
    if hasattr(model, "tokenizer"):
        model.tokenizer = pipe.tokenizer
    if hasattr(model, "scheduler"):
        model.scheduler = pipe.scheduler

    return results

def hconcat_four(a: Image.Image, b: Image.Image, c: Image.Image, d: Image.Image) -> Image.Image:
    h = a.height
    out = Image.new("RGB", (a.width + b.width + c.width + d.width, h))
    x = 0
    for im in (a, b, c, d):
        out.paste(im, (x, 0))
        x += im.width
    return out

def next_multiple_of_nv(x, nv):
    return (x + nv - 1) // nv * nv

def symmetric_pad_to_div_nv(img: Image.Image, nv: int):
    """Pad PIL image symmetrically to make (w,h) divisible by 8. Returns padded_img and pad tuple."""
    w, h = img.size
    W = next_multiple_of_nv(w, max(nv, 8))
    H = next_multiple_of_nv(h, max(nv, 8))
    pad_right = W - w
    pad_bottom = H - h
    # make it symmetric (left/top may get the extra pixel if odd)
    left = pad_right // 2
    right = pad_right - left
    top = pad_bottom // 2
    bottom = pad_bottom - top
    if left or right or top or bottom:
        # black padding is usually fine; change `fill` if you prefer white/other
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
    return img, (left, top, right, bottom)

def crop_unpad(img: Image.Image, pads):
    left, top, right, bottom = pads
    if (left or top or right or bottom):
        w, h = img.size
        return img.crop((left, top, w - right, h - bottom))
    return img

def get_reference_images(i, filenames, input_dir, NV):
    """
    Build a single reference grid image for index `i`:

    1) Get target padded size (h, w) from current NON-padded image via symmetric_pad_to_div_nv.
    2) Let f = sqrt(NV). Resize each ref to (H//f, W//f) using the original NON-padded (H, W).
    3) Manually symmetric-pad each resized ref to exactly (h//f, w//f).
    4) Tile the padded refs into a grid of size (h, w). Return the grid PIL image.

    Notes:
    - Excludes itself from sampling.
    - If there aren’t enough refs, fills with black tiles.
    - Assumes NV is a perfect square.
    - Uses your existing `symmetric_pad_to_div_nv` (and its `next_multiple_of_nv`) to get (h, w).
    """
    # 0) Current NON-padded image and sizes
    cur_img = Image.open(os.path.join(input_dir, filenames[i])).convert("RGB")
    W, H = cur_img.size

    # 1) Target padded size (h, w) using your helper (calls next_multiple_of_nv internally)
    padded_cur, _ = symmetric_pad_to_div_nv(cur_img, NV)
    w, h = padded_cur.size  # final grid size

    # 2) f = sqrt(NV), target cell size
    f = int(math.isqrt(NV))
    assert f * f == NV, f"NV={NV} must be a perfect square"
    cell_w = w // f
    cell_h = h // f

    # 3) Sample NV refs (exclude itself); fill shortfall with blanks
    candidates = [idx for idx in range(len(filenames)) if idx != i]
    if NV <= len(candidates):
        ref_indices = random.sample(candidates, NV)
    else:
        ref_indices = candidates[:] + [-1] * (NV - len(candidates))

    # 4) Compose the (h, w) grid
    grid = Image.new("RGB", (w, h), (0, 0, 0))
    k = 0
    base_w = max(1, W // f)  # avoid 0-size
    base_h = max(1, H // f)

    for r in range(f):
        for c in range(f):
            idx = ref_indices[k]
            k += 1

            if idx == -1:
                # direct blank tile at final cell size
                ref_img = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
            else:
                ref_path = os.path.join(input_dir, filenames[idx])
                ref_img = Image.open(ref_path).convert("RGB")

                # 2) resize FIRST to (H//f, W//f) from NON-padded size
                ref_img = ref_img.resize((base_w, base_h), Image.BILINEAR)

                # 3) then MANUALLY symmetric-pad to exactly (cell_w, cell_h)
                rw, rh = ref_img.size
                pad_right  = max(0, cell_w - rw)
                pad_bottom = max(0, cell_h - rh)
                left   = pad_right // 2
                right  = pad_right - left
                top    = pad_bottom // 2
                bottom = pad_bottom - top
                if left or right or top or bottom:
                    ref_img = ImageOps.expand(ref_img, border=(left, top, right, bottom), fill=0)

                # If the resized image accidentally exceeded the target (shouldn't for downsizing),
                # crop to be safe (rare edge case due to rounding).
                if ref_img.size != (cell_w, cell_h):
                    ref_img = ref_img.crop((0, 0, cell_w, cell_h))

            # 4) paste tile
            x0 = c * cell_w
            y0 = r * cell_h
            grid.paste(ref_img, (x0, y0))

    return grid

# model loads difix weights
#results = load_pipe_weights_into_model(pipe, model, report=True)

filenames = sorted(os.listdir(input_dir))

# Loop
for i, filename in enumerate(filenames):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {filename}...")
        image = Image.open(input_path).convert("RGB")
        
        # pad → run → unpad
        padded_img, pads = symmetric_pad_to_div_nv(image, NV)
        W, H = padded_img.size

        # use previous frame as reference
        if i==0:
            ref_path = os.path.join(input_dir, filenames[i+1])
        else:
            ref_path = os.path.join(input_dir, filenames[i-1])
        ref_image = Image.open(ref_path).convert("RGB")
        padded_ref_img, ref_pads = symmetric_pad_to_div_nv(ref_image, NV)
        
        if NV!=1:
            padded_ref_img = get_reference_images(i, filenames, input_dir, NV)

        padded_ref_img.save("a.png")

        # sls
        out_img = model.sample(
            padded_img,
            height=H,
            width=W,
            ref_image=padded_ref_img,
            prompt="remove degradation",
        )
        out_img = crop_unpad(out_img, pads)
        #out_img.save(output_path)


        # use previous frame as reference
        if i==0:
            ref_path = os.path.join(input_dir, filenames[i+1])
        else:
            ref_path = os.path.join(input_dir, filenames[i-1])
        ref_image = Image.open(ref_path).convert("RGB")
        padded_difix_ref_img, ref_pads = symmetric_pad_to_div_nv(ref_image, NV)
        # difix
        # Run pipeline
        output_image = pipe(
            "remove degradation",
            image=padded_img,
            ref_image=padded_difix_ref_img,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0
        ).images[0]
        # Save processed image
        output_image = crop_unpad(output_image, pads)
        #out_img.save(output_path)
        #output_image.save(output_path)

        triptych = hconcat_four(padded_ref_img, image, output_image, out_img)
        triptych.save(output_path)


print("All images processed.")


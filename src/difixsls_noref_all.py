import os
import numpy as np
from PIL import Image, ImageOps
from model import Difix
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image



# Paths
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp2" 
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_29999"
input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/old_mask_not_undist/patio_high_first/renders/render_29999"
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp3" 
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_difixsls2_noreference_29999"
output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/old_mask_not_undist/patio_high_first/renders/render_difixsls_noref_29999"

# --- INPUTS (edit these) ---
CKPT = "/home/johnny305/Documents/dfpaint/difix_sls/bs1_2gpu_pretrain/model_25001.pkl"
LORA_RANK_VAE = 4          # --lora_rank_vae (not directly used here but keep for reference)
TIMESTEP = 199             # --timestep
MV_UNET = False            # set True if you trained with --mv_unet

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

def hconcat_three(a: Image.Image, b: Image.Image, c: Image.Image) -> Image.Image:
    h = a.height
    out = Image.new("RGB", (a.width + b.width + c.width, h))
    x = 0
    for im in (a, b, c):
        out.paste(im, (x, 0))
        x += im.width
    return out

def next_multiple_of_8(x):
    return (x + 7) // 8 * 8

def symmetric_pad_to_div8(img: Image.Image):
    """Pad PIL image symmetrically to make (w,h) divisible by 8. Returns padded_img and pad tuple."""
    w, h = img.size
    W = next_multiple_of_8(w)
    H = next_multiple_of_8(h)
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


# model loads difix weights
#results = load_pipe_weights_into_model(pipe, model, report=True)

# Loop
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing {filename}...")
        image = Image.open(input_path).convert("RGB")

        # pad → run → unpad
        padded_img, pads = symmetric_pad_to_div8(image)
        W, H = padded_img.size
        
        # sls
        out_img = model.sample(
            padded_img,
            height=H,
            width=W,
            ref_image=None,
            prompt="remove degradation",
        )
        out_img = crop_unpad(out_img, pads)
        #out_img.save(output_path)

        # difix
        # Run pipeline
        output_image = pipe(
            "remove degradation",
            image=padded_img,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0
        ).images[0]
        # Save processed image
        output_image = crop_unpad(output_image, pads)
        #out_img.save(output_path)
        #output_image.save(output_path)

        triptych = hconcat_three(image, output_image, out_img)
        triptych.save(output_path)


print("All images processed.")


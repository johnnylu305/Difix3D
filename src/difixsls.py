import os
import torch.nn as nn
from PIL import Image, ImageOps
from model import Difix
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image



# Paths
#input_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp2" 
input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_29999"
#output_dir = "/home/johnny305/Documents/dfpaint/dataset/tmp3" 
output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_difixsls2_noreference_29999"

# --- INPUTS (edit these) ---
CKPT = "/home/johnny305/Documents/dfpaint/difix_sls/bs4_2gpu/model_13001.pkl"
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

results = load_pipe_weights_into_model(pipe, model, report=True)

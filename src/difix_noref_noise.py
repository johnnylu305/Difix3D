import os
import cv2
import numpy as np
from PIL import Image
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

# ------------- CONFIG -------------
INPUT_PATH = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/train_2clutter120.png"
PROMPT = "remove degradation"
OUT_DIR = "./difix_timesteps_out"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUT_DIR, "timesteps_0_999.mp4")
FPS = 15                 # change if you like
TIMESTEPS = list(range(0, 1000))   # 0..999 inclusive
# ----------------------------------

os.makedirs(FRAMES_DIR, exist_ok=True)

# Load model
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# Load input image (PIL)
input_image = load_image(INPUT_PATH).convert("RGB")
w, h = input_image.size

def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil, dtype=np.uint8)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def abs_err_bgr(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(a_bgr, b_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)      # consistent 0..255 scale
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

input_bgr = pil_to_bgr(input_image)

writer = None
frame_count = 0

for t in TIMESTEPS:
    # Run Difix at a single explicit timestep
    out_img = pipe(
        PROMPT,
        image=input_image,
        num_inference_steps=1,
        timesteps=[t],
        guidance_scale=0.0
    ).images[0]  # PIL

    out_bgr = pil_to_bgr(out_img)
    if out_bgr.shape[:2] != input_bgr.shape[:2]:
        out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_AREA)

    err_bgr = abs_err_bgr(out_bgr, input_bgr)

    # Compose row: [ input | output | error ]
    frame = np.hstack([input_bgr, out_bgr, err_bgr])  # H x (3W) x 3

    # Labels
    cv2.putText(frame, f"Input", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Output (t={t})", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "|Output - Input|", (2*w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Save frame
    os.makedirs(OUT_DIR, exist_ok=True)
    frame_path = os.path.join(FRAMES_DIR, f"frame_{t:04d}.png")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

    # Init video writer on first frame
    if writer is None:
        fh, fw = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (fw, fh))

    writer.write(frame)

if writer is not None:
    writer.release()

print(f"Saved {frame_count} frames to: {FRAMES_DIR}")
print(f"Saved video to: {VIDEO_PATH}")


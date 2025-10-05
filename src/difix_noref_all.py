import os
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

# Paths
input_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_29999"
output_dir = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_difix_noreference_29999"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load pipeline
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# Loop through all files in input_dir
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing {filename}...")

        # Load image
        input_image = load_image(input_path)

        # Run pipeline
        output_image = pipe(
            "remove degradation",
            image=input_image,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0
        ).images[0]

        # Save processed image
        output_image.save(output_path)

print("All images processed.")

from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

#input_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/train_2clutter120.png")
input_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/b.png")
#ref_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/train_2clutter120.png")
#ref_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/train_2clutter123.png")
ref_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/a.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output_refab.png")

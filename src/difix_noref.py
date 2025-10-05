import numpy as np
import matplotlib.pyplot as plt
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/train_2clutter120.png")
prompt = "remove degradation"

print(np.array(input_image))
print(np.array(input_image).shape)
plt.imshow(input_image)
plt.show()


output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output_noref.png")

plt.imshow(output_image)
plt.show()

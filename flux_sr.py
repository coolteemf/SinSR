import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
  "jasperai/Flux.1-dev-Controlnet-Upscaler",
  torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  controlnet=controlnet,
  torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load a control image
control_image = load_image(
  "/home/francois/Projects/SinSR/Market slide 3.jpeg"
)

w, h = control_image.size

# Upscale x4
control_image = control_image.resize((w * 4, h * 4))

image = pipe(
    prompt="", 
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28, 
    guidance_scale=3.5,
    height=control_image.size[1],
    width=control_image.size[0]
).images[0]
image
print("")
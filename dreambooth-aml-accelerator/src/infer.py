from diffusers import DiffusionPipeline
import sys

print("Starting inference")

model_id = sys.argv[1]

print("Setting up pipeline", model_id)

pipe = DiffusionPipeline.from_pretrained(model_id).to("cuda")
print("Pipeline loaded")

print("Generating image")
prompt =  sys.argv[3] 
image = pipe(prompt, num_inference_steps=50, guidance_scale=3.5).images[0]
print("Image generated")

image.save(sys.argv[2])
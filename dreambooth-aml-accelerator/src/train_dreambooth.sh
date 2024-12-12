echo "instance_data_dir: $1"
echo "output_dir: $2"

echo "HF_TOKEN: $HF_TOKEN"

huggingface-cli login --token $HF_TOKEN

accelerate launch --config_file /accelerate.config /diffusers/examples/dreambooth/train_dreambooth_flux.py \
--pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
--instance_data_dir="$1" \
--output_dir="$2" \
--mixed_precision="bf16" \
--instance_prompt="a photo of sks guy"  \
--resolution=512 \
--train_batch_size=1  \
--guidance_scale=1 \
--gradient_accumulation_steps=4 \
--optimizer="AdamW" \
--lr_scheduler="constant"  \
--lr_warmup_steps=0 \
--max_train_steps=15  \
--validation_prompt="A photo of sks guy in a bucket"  \
--validation_epochs=100 \
--seed="0" \
--learning_rate=1e-5 
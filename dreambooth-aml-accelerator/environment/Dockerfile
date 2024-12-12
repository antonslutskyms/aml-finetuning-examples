FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:23

COPY accelerate.config .

RUN pip install -U "huggingface_hub[cli]"

RUN git clone https://github.com/huggingface/diffusers

WORKDIR /diffusers

RUN pip install -e .

WORKDIR /diffusers/examples/dreambooth

RUN pip install -r requirements_flux.txt

RUN pip install prodigyopt

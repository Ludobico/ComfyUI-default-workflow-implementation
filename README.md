## ComfyUI-default-workflow-implementation

[![ko](https://img.shields.io/badge/lang-ko-green.svg)](./README.ko.md)

![alt text](static/default_workflow.png)

This is an example implementation of the default workflow from [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The implementation is designed with reference to the [Diffusers](https://github.com/huggingface/diffusers) library, and the input and output formats are compliant with ComfyUI's standards.

## Supported Nodes

- Load Checkpoint
- CLIP Text Encoder (Prompt)
- Empty Latent Image
- Ksampler
- VAE Decode
- Save Image
- Preview Image

## Installation

Use the commands below to install and run the project.  
PyTorch must be installed with CUDA support to enable GPU acceleration.

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python default_workflow.py
```

Using the `Save Image` node saves the generated image as `output - ComfyUI_{number}.png`.

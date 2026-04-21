# BiReG
BiReG: Adaptive Region Planning for Training-Free Bilingual Text-to-Image Generation  
**Zhuang Ye**，**……**，**……**
## Overview
BiReG is a training-free region-guided framework for bilingual text-to-image generation. It addresses the challenge of controllable generation under both English and Chinese prompts, where existing methods often rely on fixed English templates and struggle to capture Chinese semantic structures.  
BiReG leverages an LLM-driven region planner to dynamically infer spatial layouts from natural language prompts, enabling structured and flexible region control without predefined templates. By integrating region-aware prompts into the diffusion process, BiReG improves both spatial consistency and semantic alignment.  

This repository is organized into two parts: 
- **Demo Mode**: Provides predefined prompt-region pairs to illustrate the core idea of region-guided generation and help users understand the framework behavior.
- **Full Pipeline**: Supports end-to-end generation from user-defined prompts, including region planning and diffusion-based synthesis.

## Demo Usage (Stage 1: Understanding BiReG)

The demo mode provides predefined prompt-region pairs to illustrate how BiReG performs region-guided generation.

### 1. Environment Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```
### 2. Prepare Model Weights
Please download and place the Kolors model weights under:

```bash
/root/autodl-tmp/weights/Kolors/
```
The directory should include:

- text_encoder
- vae
- scheduler
- unet

Make sure the tokenizer file (e.g., tokenizer.model) is also available.

### 3.Run Demo
Execute the following command:
```bash
python demo_infer.py
```

### 4.Output
The generated image will be saved in the current directory:
```bash
traveler_cat_fire.png
```
### 5. What This Demo Shows
- How prompts are decomposed into multiple regions
- How spatial layouts are controlled via split_ratio
- How region-specific prompts affect generation
This demo helps users understand how BiReG performs region-guided generation before using the full pipeline.

# BiReG: Adaptive Region Planning for Training-Free Bilingual Text-to-Image Generation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19853901.svg)](https://doi.org/10.5281/zenodo.19853901)

Official implementation of **BiReG**, a training-free framework for bilingual (Chinese-English) text-to-image generation with **LLM-driven adaptive region planning**.


<p align="center">
  <img src="/assets/figure2效果对比图-2.svg" width="1200"/>
</p>

## 🔍 Overview

BiReG addresses a fundamental limitation in controllable text-to-image generation:

> Existing region-guided methods rely on fixed, English-centric templates, which fail to capture the semantic structure of Chinese prompts, especially in complex compositional scenarios.

To solve this, BiReG introduces an **LLM-driven adaptive region planning mechanism**, which:

- parses bilingual prompts (Chinese & English)
- infers spatial layout dynamically
- generates region-specific prompts
- injects them into diffusion models without retraining

---

## 🚀 Key Contributions
<table>
<tr>
<td width="45%">

- **Training-Free Framework**
  - No finetuning required
  - Compatible with existing diffusion backbones (Kolors, SDXL)

- **Adaptive Region Planning**
  - Dynamic layout inference (instead of fixed templates)
  - Supports hierarchical spatial structures

- **Bilingual Semantic Understanding**
  - Handles Chinese and English prompts directly
  - Preserves modifier–noun dependencies in Chinese

- **Region-Guided Diffusion Control**
  - Injects regional prompts into cross-attention layers
  - Improves spatial consistency and object alignment
</td>
<td width="55%">
<img src="assets/Figure1-翻译流程vs自适应流程示意.svg" width="100%">
</td>

</tr>
</table>

---

## 🧠 Framework Pipeline
<p align="center">
  <img src="assets/figure3-overview.svg" width="1200"/>
</p>

```text
Input Prompt (Chinese / English)
        ↓
Language Detection
        ↓
Prompt Structuring
        ↓
LLM Planner
        ↓
Structured Output:
    - Final split ratio
    - Regional prompt
        ↓
Region-Guided Diffusion
        ↓
Generated Image
```
## Repository Structure
```text
BiReG/
├── demo_infer.py
├── full_infer.py
├── planner.py
├── RegionalKolorsDiffusion_xl.py
├── template/
│   ├── template_zh.txt
│   ├── template_en.txt
├── config/
│   ├── api_config_example.json
├── outputs/
│   ├── demo/
│   ├── full/
```
## ⚙️ Installation
```text
git clone https://github.com/YeZhuang-Joe/BiReG.git
cd BiReG
pip install -r requirements.txt
```
## 📥 Model Preparation
Prepare pretrained Kolors weights:
```text
<MODEL_ROOT>/Kolors/
├── text_encoder/
├── vae/
├── scheduler/
├── unet/
```
Set environment variable:
```text
export KOLORS_PATH=/path/to/Kolors
```
## 🔑 API Configuration
Create:
```text
config/api_config.json
```
Example:
```text
{
  "deepseek_api_key": "your_key",
  "openai_api_key": "your_key"
}
```
## 🔍 Demo (Stage 1:Fixed Region-Controlled Generation)
We provide a set of **fixed demo cases** to illustrate the effectiveness of BiReG in spatially controllable text-to-image generation.

Unlike stochastic prompt-based generation, each demo case includes:
- a fixed input prompt
- a predefined spatial layout (split ratio)
- region-specific prompts
- a fixed inference configuration
This ensures **fully reproducible results**.

---
### 📦 Available Demo Cases

Run the following command to list all demo cases:
```bash
python demo_infer.py --list
```

Example output:
```text
[INFO] Available demo cases:
  - palace_two_maids: Dual-subject palace corridor scene with explicit foreground-background separation.
  - scholar_squirrel_bamboo: 
  - maoniu: 
  - yumin: 
```
---

### ▶️ Run Demo Case
To generate an image:
```bash
python demo_infer.py --case palace_two_maids

```
### 📁 Output Structure
Generated results are automatically saved to:
```text
outputs/demo/
```
Each run produces:
```text
<case_name>_<timestamp>.png   # generated image
<case_name>_<timestamp>.json  # metadata
```
### 🧾 Metadata (Reproducibility)
Each .json file records:
- input prompt
- split ratio (layout)
- regional prompts
- inference configuration (steps, resolution, guidance scale, seed)
- output path
### 🧠 Example Case: 
<p align="center">
  <img src="/assets/宫女-背影.png" width="1200"/>
</p>
<p align="center">
  <img src="/assets/松鼠-读书人.png" width="1200"/>
</p>

## 🚀 Full Pipeline (Stage 2: Method Demonstration)
---
Unlike Stage 1, this stage demonstrates the core mechanism of BiReG.
---
### 🧠 What This Stage Shows
- LLM-based semantic parsing
- adaptive layout generation
- region-conditioned diffusion
---
### ▶️ Example (Chinese)
```text
python full_infer.py \
--prompt "上方是天空和远山，下方左边是旅人，下方右边是猫" \
--planner deepseek
```
LLM Output
```text
Final split ratio:
0.3,1;0.7,0.5,0.5
Regional Prompt:
天空高远，远山层叠 BREAK
左下角旅人，背包，站立 BREAK
右下角猫，细节清晰
```
🌍 English Example
```text
python full_infer.py \
--prompt "Sky on top, traveler bottom left, cat bottom right" \
--planner deepseek
```
### ⚠️ Note on LLM Variability
- Outputs may vary slightly across runs
- Structure remains consistent
- Semantics preserved

## 📄 Paper & Citation

This manuscript has been submitted to *The Visual Computer* for review.
If you find this code useful for your research, please consider citing the corresponding paper once it becomes publicly available. 

---

## 📬 Contact
For questions, please contact the authors.
- **Zhuang Ye** – yj20242054@stud.tjut.edu.cn  
We welcome discussions and collaborations.

# BiReG: Training-Free Adaptive Region Planning for Bilingual Text-to-Image Generation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19853901.svg)](https://doi.org/10.5281/zenodo.19853901)

Official implementation of **BiReG**, a training-free framework for bilingual (Chinese-English) text-to-image generation with **LLM-driven adaptive region planning**.


<p align="center">
  <img src="/assets/figure2效果对比图-2.svg" width="1200"/>
</p>

## Main-experiment reproduction

The [main-experiment guide](experiments/main_experiment/README.md) provides the frozen EN1920 and ZH150 prompt collections, six-pipeline generation and evaluation entries, archived automatic scores, statistical reproduction scripts, and image-download information. The main experiments cover 1,920 English and 150 Chinese-source prompts, with 37,260 retained images across six pipelines. Configuration, environment, and validation records accompany the corresponding entries. Use this guide for the retained main-experiment inputs and results; the demonstrations below illustrate method usage.


## 🔍 Overview

BiReG is a training-free framework for Chinese–English text-to-image generation. It connects language-aware prompt structuring, adaptive region planning, and generation control within a unified bilingual workflow.

Using language-specific structuring templates, an LLM converts a user prompt into regional descriptions and a spatial layout. The resulting plan guides a pretrained Kolors model through regional and global conditioning, without updating the diffusion backbone parameters.

## 🚀 Key Contributions
<table>
<tr>
<td width="45%">

- **Unified Bilingual Workflow**
  - Supports Chinese and English prompts through language-specific structuring templates.
  - Organizes entities, attributes, and relationships into regional descriptions and spatial layouts.

- **Adaptive Region Planning**
  - Uses LLM reasoning to infer region arrangements and split ratios from user descriptions.
  - Coordinates regional content with the planned spatial organization.

- **Training-Free Generation**
  - Guides a pretrained Kolors model without updating diffusion backbone parameters.

- **Regional–Global Conditioning**
  - Combines regional and global cross-attention outputs according to the planned layout.
  - Uses a fusion weight to control their relative contributions.
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

| Directory | Contents |
| --- | --- |
| [`generation/`](generation/README.md) | Chinese–English prompt-to-image workflow, with setup instructions and environment records. |
| [`experiments/main_experiment/`](experiments/main_experiment/README.md) | Frozen EN1920 and ZH150 prompts, generation and evaluation entries, archived scores, statistical scripts, and image-download information. |
| [`experiments/human_evaluation/`](experiments/human_evaluation/README.md) | Human-evaluation design, anonymized ratings, statistical scripts, and image-material access information. |
| [`experiments/planner_comparison/`](experiments/planner_comparison/README.md) | Planner-comparison prompts, archived evaluation data, and statistical reproduction scripts. |
| [`experiments/efficiency/`](experiments/efficiency/README.md) | Planning and generation timing records and summary reproduction scripts. |
| [`experiments/prompt_length_statistics/`](experiments/prompt_length_statistics/README.md) | Prompt-length statistics and the corresponding analysis script. |
| [`data/development_prompts/`](data/development_prompts/README.md) | The curated collection of 50 Chinese–English development prompt pairs and accompanying notes. |
| [`paper_figures/`](paper_figures/README.md) | Original images and prompt records for the corresponding paper figures. |

## ⚙️ Installation and Environment Records

```bash
git clone https://github.com/YeZhuang-Joe/BiReG.git
cd BiReG
```

Choose the entry point appropriate for your task:

- [Prompt-to-image workflow](generation/README.md): language routing, API-based regional planning, and Kolors generation.
- [Frozen main-experiment generation](experiments/main_experiment/generation/README.md): six-pipeline generation using the retained prompts, configurations, and saved plans.

Follow the environment, model preparation, and validation instructions in the corresponding guide. The entry points have separate recorded environments. The root `requirements.txt` is a historical dependency list, not a verified clean-install lock for all pipelines. The exact source revision of the recorded Diffusers development build remains to be identified.

## 🚀 Prompt-to-Image Workflow

The reusable generation entry supports Chinese and English prompts through language routing, API-based regional planning, and Kolors generation. See the [generation guide](generation/README.md) for environment, model, and API setup.

After completing that setup, run the following commands from the repository root:

```bash
cd generation

# English example
python -m bireg --prompt "A red cup to the left of a blue bowl." --output outputs/demo_en

# Chinese example
python -m bireg --prompt "木桌上，左边是一个红色陶瓷杯，右边是一个蓝色玻璃碗。" --output outputs/demo_zh
```

Each run saves the generated image, the accepted regional plan, planning records, and generation metadata. For generation using the retained main-experiment prompts and saved plans, follow the [main-experiment generation guide](experiments/main_experiment/generation/README.md).

## 📄 Paper & Citation

This manuscript has been submitted to *The Visual Computer* for review.
If you find this code useful for your research, please consider citing the corresponding paper once it becomes publicly available. 

---

## License

Author-created BiReG software contributions are licensed under the [Apache License 2.0](LICENSE). Third-party code retains its original license conditions and copyright notices; see [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The software license does not grant additional permissions to external model weights, benchmark datasets, generated images, or API services, which remain subject to their applicable terms.


## 📬 Contact
For questions, please contact the authors.
- **Zhuang Ye** – yj20242054@stud.tjut.edu.cn  
We welcome discussions and collaborations.

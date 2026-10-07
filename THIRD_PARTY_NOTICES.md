# Third-party code and license scope

## BiReG software

Copyright (c) 2026 BiReG contributors. Author-created BiReG software contributions are licensed under the Apache License, Version 2.0; see the root `LICENSE`. Third-party code and its derivatives retain applicable upstream license conditions and copyright notices. A root license does not replace licenses attached to third-party files or materials.

The following inventory was checked against BiReG commit `3e20dbb4ea6145c26ac8c651320d4fd024c86efc` on 2026-10-07. Paths identify included copies or adapted components; they do not assert that all copies are byte-identical to the latest upstream source.

## Included code

| Component | Included locations | Upstream source | License text |
| --- | --- | --- | --- |
| RPG regional diffusion, regional attention, and layout utilities | Regional-generation code including root `RegionalKolorsDiffusion_xl.py`, `cross_attention1.py`, and `matrix.py`; archived counterparts under `generation/vendor/historical_kolors/`; regional files under `experiments/main_experiment/generation/vendor/en/native_source/`, `vendor/regional/`, and `vendor/zh_baselines/native_source/` | [RPG-DiffusionMaster](https://github.com/YangLing0818/RPG-DiffusionMaster), upstream commit checked: `c92a8d1a7495e641196d16c2203bd6af427a28ea` | MIT, Copyright (c) 2024 Ling Yang; [full text](licenses/RPG-MIT.txt). Diffusers-derived portions also retain their existing Apache-2.0 headers. |
| Kolors pipelines and ChatGLM configuration/model/tokenizer code | Root `configuration_chatglm.py`, `modeling_chatglm.py`, `tokenization_chatglm.py`, and the regional Kolors pipeline; corresponding copies under `generation/vendor/historical_kolors/` and `experiments/main_experiment/generation/vendor/regional/` | [Kolors](https://github.com/kwai-kolors/Kolors), upstream commit checked: `038818d244ed103056abd10f429729a26af4d239`; [ChatGLM3](https://github.com/zai-org/ChatGLM3) for underlying model-code attribution | Apache-2.0; [Kolors text](licenses/Kolors-Apache-2.0.txt) and [ChatGLM3 text](licenses/ChatGLM3-Apache-2.0.txt). Model weights have separate terms. |
| Hugging Face Diffusers pipeline/attention code | Files retaining Hugging Face copyright and Apache-2.0 headers, including the regional diffusion pipelines and Diffusers-derived RAGD files | [Diffusers](https://github.com/huggingface/diffusers) | Apache-2.0; [full text](licenses/Diffusers-Apache-2.0.txt). Preserve existing per-file copyright notices. |
| RAG-Diffusion / RAGD baseline | `experiments/main_experiment/generation/vendor/ragd/RAG_pipeline_flux.py`, `RAG_transformer_flux.py`, `cross_attention.py`, and `matrix.py` | [RAG-Diffusion](https://github.com/NJU-PCALab/RAG-Diffusion), upstream commit checked: `2acfbd434aa775effc95baf7c765edb3bba91154` | MIT, Copyright (c) 2024 NJU-PCALab; [full text](licenses/RAG-Diffusion-MIT.txt). Diffusers-derived files additionally retain their Apache-2.0 headers and Black Forest Labs / Hugging Face attribution. |

The root and `generation/vendor/historical_kolors/` copies of `configuration_chatglm.py`, `modeling_chatglm.py`, and `tokenization_chatglm.py` match the corresponding Git blob hashes of the checked Kolors source. The `vendor/regional/modeling_chatglm.py` copy has a different blob hash and is not represented as an unchanged upstream copy.

## Other software dependencies

Installed dependencies such as PyTorch, Transformers, xFormers, and other packages retain their own licenses. The BiReG software license does not relicense external packages. Preserve relevant dependency notices when redistributing packaged dependencies.

## Model weights, benchmark materials, and images

The Apache-2.0 license for author-created BiReG software does not grant additional permissions to Kolors, SDXL, FLUX.1-dev, or evaluator model weights. Obtain weights from their original sources and follow the corresponding model licenses.

Benchmark prompts, translations, annotations, evaluation outputs, human-evaluation materials, and paper images are outside this software-license declaration. Their reuse remains subject to applicable source terms and any material-specific permission. The included UniGenBench dataset license is available at `experiments/main_experiment/evaluation/references/UniGenBench_License.txt` (CC BY 4.0). This document does not assign a new blanket data or image license.

## Redistribution and modifications

Keep original copyright notices, license headers, and the relevant complete license texts with redistributed source. Files modified from Apache-2.0 sources must carry a prominent modification notice as required by that license. Preserve the original provenance of frozen source snapshots; any notice-bearing replacements or patches should be recorded separately so their hashes are not confused with the original archived files.

This inventory documents the components identified above; it is not a claim that every historical file is unchanged, that all model weights share one license, or that public access alone grants reuse rights to every asset.

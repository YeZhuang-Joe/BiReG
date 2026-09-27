# Environment provenance

The predecessor workflow was run on the user's existing AutoDL `kolors` environment. This distribution preserves its backend and templates; a GPU smoke test of the reorganized entry is still required.

| Component | Observed version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.2.1+cu118 |
| torchvision | 0.17.2 |
| Diffusers | 0.33.0.dev0, installed as an egg; Git revision unavailable |
| transformers | 4.42.4 |
| accelerate | 0.27.2 |
| xformers | 0.0.25+cu118 |
| NumPy | 1.24.3 |
| Pillow | 10.4.0 |
| sentencepiece | 0.1.99 |
| safetensors | 0.4.1 |
| huggingface-hub | 0.30.1 |
| opencv-python | 4.8.1.78 |

These are observations, not a tested clean-install lockfile. Do not replace the working Diffusers build by installing an arbitrary release with a similar number. Before claiming a fresh installation is reproducible, archive/identify that build and test a clean environment. This package does not download dependencies or weights.

The local checkpoint must contain `text_encoder/`, `unet/`, `vae/`, `scheduler/`, and `model_index.json`. Loader configuration hashes and weight-file inventory are recorded; weight contents are not fully hashed by this entry. CUDA and xformers are required. CPU offload is enabled. Set `OMP_NUM_THREADS=1` if the previous environment set it to the invalid value 0.

```bash
python tools/check_environment.py --model-path /root/autodl-tmp/weights/Kolors --output outputs/environment.json
```

This probes imports and loads the tokenizer; it does not load all model weights, generate images or call the API. Use a new output path if an existing environment record differs.

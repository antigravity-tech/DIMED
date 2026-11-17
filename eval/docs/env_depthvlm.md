# Environment for DepthVLM-4B

DepthVLM needs `transformers 5.2.0` (the version the checkpoint was saved with), so it runs in its own environment.
What was used (Python 3.12, torch 2.5.0 + CUDA 12.4):

```bash
python -m venv --system-site-packages depthvlm_env      # on top of an environment that already has torch 2.5.0
./depthvlm_env/bin/pip install "transformers==5.2.0" "accelerate==1.6.0" "qwen-vl-utils==0.0.14" \
    safetensors einops click scipy matplotlib tqdm opencv-python-headless
./depthvlm_env/bin/pip install --no-deps \
    "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"
```

Then `PY_DEPTHVLM=$PWD/depthvlm_env/bin/python bash run.sh depthvlm 0`.

Notes
- `utils3d`, `utils3d-moge`, `pipeline` are the commits pinned by MoGe's requirements; MoGe's data loader imports them.
- The adapter turns off the cuDNN scaled-dot-product-attention backend (`torch.backends.cuda.enable_cudnn_sdp(False)`);
  with torch 2.5.0 it fails with "No execution plans support the graph" on this model.
- The checkpoint is about 9.7 GB (`model.safetensors`).

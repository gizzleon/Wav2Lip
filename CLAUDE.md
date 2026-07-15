# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The open-source Wav2Lip implementation from the ACM MM 2020 paper *A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild*. It lip-syncs a video/image to an arbitrary audio track. The README also documents sync.so's hosted commercial API (`syncsdk`), which is a separate product — none of that API code lives in this repo.

License note: this code and the LRS2-trained weights are research/personal use only; commercial use is prohibited.

## Setup

The environment is managed by **uv** — [pyproject.toml](pyproject.toml) + [.python-version](.python-version) + `uv.lock` are the source of truth. `uv sync` builds `.venv`; run scripts with `uv run python <script>.py`.

**[requirements.txt](requirements.txt) is stale and unusable — do not `pip install -r` it.** It is kept only as a record of the authors' original 2020 environment (Python 3.6, `torch==1.1.0`, `librosa==0.7.0`). Neither Python 3.6 (uv's oldest is 3.8) nor `torch==1.1.0` (no macOS arm64 wheels ever built) can be installed on Apple Silicon. `pyproject.toml` is the working replacement; edit that instead.

Why the versions in `pyproject.toml` are what they are — this chain is load-bearing and easy to break:
- **`librosa==0.9.2`** — 0.10 made `librosa.filters.mel()` keyword-only, and [audio.py:100](audio.py#L100) calls it positionally. 0.9.2 is the last release that runs the code unpatched (it emits a `FutureWarning`). Upgrading past it requires patching `_build_mel_basis` to `mel(sr=..., n_fft=...)`.
- **`numba<0.57`** — librosa 0.9.2's JIT backend. This is what caps Python at **<3.11** and numpy at **<1.24**, not librosa itself.
- **`setuptools<81`** — librosa 0.9.2 imports `pkg_resources`, removed in setuptools 81. Without this pin `import librosa` dies at `ModuleNotFoundError: pkg_resources`, since numba pulls in whatever setuptools is newest.
- **`opencv-contrib-python` only** — the original requirements.txt lists *both* `opencv-python` and `opencv-contrib-python`, which install competing `cv2` modules over each other. contrib is a superset; keep just it.
- **`scipy`** — used directly by `audio.py` and `inference.py` but absent from the original requirements.txt; it only ever arrived transitively via librosa.

`ffmpeg` must be on PATH (present via Homebrew); it is shelled out to for audio extraction and final muxing.
- The S3FD face detector weights must be at `face_detection/detection/sfd/s3fd.pth`. `preprocess.py` hard-fails at import if missing; `inference.py` fails later at detection time.
- `checkpoints/`, `filelists/`, `temp/`, `results/` are committed as empty placeholders (README-only). `temp/` is not optional — inference writes `temp/temp.wav` and `temp/result.avi` there mid-run.
- `.gitignore` excludes `filelists/*.txt`, `*.pth`, `*.jpg`, `*.mp4`, so datasets, filelists, and weights are all supplied out-of-band.

## Commands

There is no test suite, linter, or build step. Everything is a `uv run python <script>.py` entry point run from the repo root — `get_image_list()` in [hparams.py](hparams.py) opens `filelists/{split}.txt` as a **relative** path, so training breaks from any other cwd.

```bash
uv sync   # create/update .venv from uv.lock
```

Inference:
```bash
uv run python inference.py --checkpoint_path <ckpt.pth> --face <video.mp4|image.jpg> --audio <audio.wav|any-ffmpeg-readable>
# → results/result_voice.mp4
```

Preprocess LRS2 into frames + 16kHz wavs (**requires CUDA** — device is hardcoded `cuda:{id}`):
```bash
uv run python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/ --ngpu 1 --batch_size 32
```

Train — **strictly two stages, in this order**:
```bash
# 1. Expert lip-sync discriminator (SyncNet). Must converge to eval loss ~0.25.
uv run python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <dir>

# 2a. Generator, L1 + sync loss only (<1 day)
uv run python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <dir> --syncnet_checkpoint_path <syncnet.pth>

# 2b. Generator + visual quality discriminator (~2 days, better visual quality, slightly worse sync)
uv run python hq_wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <dir> --syncnet_checkpoint_path <syncnet.pth>
```
Both training scripts take `--checkpoint_path` to resume; `hq_wav2lip_train.py` additionally takes `--disc_checkpoint_path`. Training never terminates on its own (`nepochs` is effectively infinite) — stop manually when eval sync loss reaches ~0.2 and stops improving.

Evaluation lives in `evaluation/` and **requires a separate virtualenv** — it depends on the upstream `syncnet_python` repo, whose dependency versions conflict with this one's. See [evaluation/README.md](evaluation/README.md). Its scripts are also unconditionally `.cuda()`.

### What actually runs on this machine (Apple Silicon, no CUDA)

The code only ever tests `torch.cuda.is_available()`; it has no concept of MPS, which is the only accelerator here. So:
- **Inference works** — `inference.py` falls back to `device = 'cpu'` cleanly. It runs on CPU only, and will be slow. MPS is available (`torch.backends.mps.is_available()` is True) but unused; nothing wires it up.
- **`preprocess.py` cannot run** — [preprocess.py:33](preprocess.py#L33) hardcodes `device='cuda:{}'`, with no fallback.
- **`hq_wav2lip_train.py` cannot run** — [models/wav2lip.py:172](models/wav2lip.py#L172) hardcodes `.cuda()` inside `perceptual_forward`, which is reached whenever `disc_wt > 0` (default `0.07`). `wav2lip_train.py` and `color_syncnet_train.py` respect the device variable and would technically run on CPU, but training this at CPU speed is not realistic.

Treat this as a CUDA box's repo: preprocessing and training belong on a CUDA machine, and this checkout is practical for inference and code work.

## Architecture

**Two-model design.** `SyncNet_color` ([models/syncnet.py](models/syncnet.py)) is a frozen *expert* — trained once on real video, then used only as a loss function for the generator. It is never fine-tuned during generator training (`requires_grad = False` on all params). This is the paper's core idea: a pre-trained, frozen sync critic produces far better lip-sync than a jointly-trained one. Training on a new dataset therefore means retraining SyncNet on that dataset first.

**Generator** ([models/wav2lip.py](models/wav2lip.py)) is a U-Net: the face encoder downsamples 96×96 → 1×1 collecting skip features, the audio encoder maps a mel chunk to a 512-d embedding, and the decoder starts from the audio embedding and concatenates skips on the way back up. Input face is **6 channels**: the target frame with its lower half zeroed, concatenated with a reference frame (a different random frame of the same speaker at train time; the same frame at inference).

**Everything operates on the lower half of the face only** for loss purposes — `SyncNet` (15 channels = 5 frames × RGB, lower half), `Wav2Lip_disc_qual` (`get_lower_half`). The generator outputs a full 96×96 face, but only the mouth region is judged.

**Tight temporal coupling.** `syncnet_T = 5` frames ↔ `syncnet_mel_step_size = 16` mel steps ↔ `fps = 25` ↔ 80 mel frames/sec. These constants are duplicated across [color_syncnet_train.py](color_syncnet_train.py), [wav2lip_train.py](wav2lip_train.py), [hq_wav2lip_train.py](hq_wav2lip_train.py), and [inference.py](inference.py) (as `mel_step_size`) rather than centralized. The 80/fps ratio appears as `mel_idx_multiplier` in inference and inline in `crop_audio_window`. Changing dataset FPS breaks these relationships in several places at once — the README explicitly warns against it.

**Loss weighting is dynamic.** `hparams.syncnet_wt` starts at `0.0` and is flipped to `0.03` at runtime by the training loop once eval sync loss drops below 0.75 (see `eval_model` in the train scripts). Starting with sync loss enabled hurts convergence. In `hq_wav2lip_train.py` the total is `syncnet_wt * sync + disc_wt * perceptual + (1 - syncnet_wt - disc_wt) * l1`, so the three weights are coupled and must stay under 1.

**Config.** [hparams.py](hparams.py) holds mel/STFT params and training hyperparameters, shared by preprocessing, training, and (indirectly) inference. Note `inference.py` hardcodes `args.img_size = 96` instead of reading hparams — changing `img_size` requires editing both.

**Checkpoints** are dicts of `state_dict`/`optimizer`/`global_step`/`global_epoch`. Loaders strip a `module.` prefix from keys to tolerate weights saved under `DataParallel`. Set `hparams.save_optimizer_state = False` to halve checkpoint size when resume isn't needed.

**Vendored code:** `face_detection/` is a trimmed copy of [face-alignment](https://github.com/1adrianb/face-alignment) (S3FD detector only); [audio.py](audio.py) is the mel pipeline from `deepvoice3_pytorch`. Prefer leaving both alone — they're synced with upstream, not authored here.

## Inference gotchas worth knowing before debugging output quality

These are user-facing knobs, not bugs — most "bad output" reports resolve to one of them:
- `--pads 0 20 0 0` — the detected box often clips the chin; bottom padding is the most common fix.
- `--nosmooth` — detections are smoothed over a 5-frame window by default (`get_smoothened_boxes`), which causes a dislocated or doubled mouth when the head moves fast.
- `--resize_factor` — the model is trained on low-res faces, so 720p often beats 1080p.
- `--box` — bypasses detection entirely with a fixed box; last resort for undetected faces.
- Face detection batch size auto-halves on CUDA OOM and retries; it does not auto-recover from a too-large frame at batch size 1.
- A NaN mel raises explicitly — usually a TTS-generated wav with digital silence; add epsilon noise.

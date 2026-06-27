# Georgian Sign Language Interpreter

Prototype pipeline for collecting Georgian Sign Language samples from a webcam, training a small classifier, and running live text inference.

This is a controlled-vocabulary prototype, not a general sign-language translator.

## Local Environment

This repo uses a project-local Python runtime:

```powershell
.\.runtime\python311\python.exe --version
.\.venv\Scripts\python.exe --version
```

The global Python install is not used.

Install this package into the local venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Commands

Record samples for a label. For the high-accuracy body-relative model, prefer 30-50 samples per sign:

```powershell
.\.venv\Scripts\python.exe -m gsl_interpreter record --label "გამარჯობა" --samples 40 --signer giorgi
```

Train a classifier:

```powershell
.\.venv\Scripts\python.exe -m gsl_interpreter train --model-out models/classifier.pkl
```

Training uses a PyTorch GRU sequence classifier. If CUDA is available, it trains on the NVIDIA GPU.

Run live inference:

```powershell
.\.venv\Scripts\python.exe -m gsl_interpreter infer --model models/classifier.pkl
```

Inference runs the PyTorch sequence model and per-sequence motion feature construction on CUDA when available. MediaPipe hand/body tracking and OpenCV camera/display still run on CPU in this prototype; the default inference camera settings request a lower-latency 960x540 stream and faster MediaPipe tracking to reduce lag.

If tracking accuracy drops, raise the MediaPipe pose model complexity:

```powershell
.\.venv\Scripts\python.exe -m gsl_interpreter infer --model models/classifier.pkl --tracking-complexity 1
```

If the camera is still laggy, lower the requested camera size:

```powershell
.\.venv\Scripts\python.exe -m gsl_interpreter infer --model models/classifier.pkl --width 640 --height 360
```

The live inference window now builds a sentence instead of locking after one word:

- Each accepted sign is appended to the on-screen sentence.
- The terminal prints the full sentence after each accepted sign.
- The inference view shows a clean camera feed with a right-side operator console for sentence text, recognition status, capture progress, recent word chips, actions, and last-confidence metadata.
- Use the `Reset` button or `R` to clear the sentence.
- Use the `Undo` button, `U`, or Backspace to remove the last word.
- If you later train punctuation/control labels, `წერტილი`/`period`, `მძიმე`/`comma`, `კითხვის ნიშანი`/`question mark`, and `ძახილის ნიშანი`/`exclamation mark` are handled specially.

## Data Contract

- Raw samples: `data/raw/<label>/<signer_id>_<timestamp>.npy`
- Sample shape: `(30, 351)`, dtype `float32`
- Labels: `data/labels.json`, UTF-8 JSON, `{ "label": id }`
- Model artifact: `models/classifier.pkl`, joblib bundle with `{model, label_map, feature_version}`
- Current model artifact is a PyTorch sequence bundle with `{model_type, model_config, state_dict, label_map, feature_version, start_templates}`
- Each frame stores hand shape, hand position relative to the shoulders/body frame, and pose landmarks. The PyTorch model also receives per-frame velocity features.

## Agent Coordination

Codex and Claude coordinate through `AGENT_NOTES.md`.

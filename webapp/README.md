# VQA Interactive Web Demo

An interactive demo for the Visual Question Answering (VQA v2.0) project.
Ask any question about an image and compare 4 neural architectures side-by-side in real time.

---

## Quick Start

```bash
# From the project root
pip install flask
python webapp/app.py
```

Open **http://localhost:5000** in your browser.
Model loading takes ~8–15 seconds on first startup.

### Access from Another Machine (Tailscale)

```bash
# On the host machine — get your Tailscale IP
tailscale ip -4           # e.g. 100.x.x.x

# Start the app (already binds to 0.0.0.0)
python webapp/app.py
```

On the remote machine, open: **http://100.x.x.x:5000**

No port forwarding or firewall rules needed — Tailscale handles routing.

**Keep running after closing terminal:**

```bash
# Simple background run
nohup python webapp/app.py > webapp/app.log 2>&1 &
echo $! > webapp/app.pid

# Stop later
kill $(cat webapp/app.pid)
```

---

## How to Use

1. **Upload an image** — drag and drop onto the left panel, or click a COCO val2014 thumbnail
2. **Type a question** in the text box (see examples below)
3. **Select models** — tick any combination of A / B / C / D
4. **Choose decode mode** — Greedy, Beam Search, or Both
5. Click **Run Inference**

Results appear as per-model cards with three tabs each: **Attention**, **Confidence**, and **Top-5**.

---

## The 4 Models

| Model | Image Encoder | Decoder | Notes |
|-------|--------------|---------|-------|
| **A** | Scratch CNN (5 conv blocks) | LSTM, no attention | Fastest baseline |
| **B** | ResNet101 (pretrained ImageNet) | LSTM, no attention | Best efficiency |
| **C** | Scratch CNN (49 spatial regions) | LSTM + Bahdanau Attention | Attention, no pretraining |
| **D** | ResNet101 (pretrained, 49 regions) | LSTM + Bahdanau Attention | Best overall |

**Published results (n = 88,488 val samples):**

| Model | BLEU-4 | METEOR | ROUGE-L | Exact Match |
|-------|--------|--------|---------|-------------|
| A | 0.0915 | 0.3117 | 0.3828 | 2.83% |
| B | 0.1127 | 0.3561 | 0.4237 | 4.07% |
| C | 0.0988 | 0.3271 | 0.3971 | 4.18% |
| **D** | **0.1159** | **0.3595** | **0.4270** | **5.88%** |

---

## Understanding the Metrics

### Confidence Score

The **confidence** of a token is the softmax probability the model assigns to its top prediction at that decode step — a number between 0 and 1.

- **High confidence (≥ 0.8):** The model is very certain about this word. Shown in green.
- **Medium confidence (0.4–0.8):** Some uncertainty. Shown in yellow.
- **Low confidence (< 0.4):** The model is guessing between several plausible words. Shown in red.

The **average confidence** shown on the card is the mean across all generated tokens. A high average means the model produced the answer decisively; a low average suggests the image–question pair is ambiguous or out-of-distribution.

> **Example:** For "What color is the car?", Model D might generate "red" with confidence 0.94 — the image has a single dominant red car and the feature is unambiguous. For "Why is the person smiling?", confidence will be much lower (~0.3) because the answer requires reasoning beyond visual features.

---

### Attention Heatmap (Models C and D only)

At each decoder step, the Bahdanau attention mechanism assigns a weight to each of the **49 spatial regions** (7 × 7 grid) of the image. The heatmap shows which regions the model "looked at" when generating each word.

- **Bright/hot regions (yellow–white)** — high attention weight, the model focused here
- **Dark regions (purple–black)** — low attention weight, mostly ignored

Use the **word buttons** or the **slider** below the heatmap to step through each generated token and watch the model's focus shift.

The **Average Heatmap** blends all per-token attention maps, giving an overall picture of which image regions mattered most for the full answer.

---

### Attention Entropy

Entropy measures **how spread out** the attention weights are across the 49 regions. It is computed as:

```
H = −Σ αᵢ · log(αᵢ)    for i = 1…49
```

where αᵢ is the attention weight on region i.

| Entropy | Meaning | What it looks like |
|---------|---------|-------------------|
| **Low entropy** (e.g. 0.5–1.5) | **Focused** — the model concentrated most of its attention on 1–3 regions | A small bright spot on a dark background |
| **High entropy** (e.g. 3.0–3.9) | **Diffuse** — the model spread attention roughly equally across many regions | A nearly uniform warm glow across the whole image |

**Why does this matter?**

- **Focused attention** on the correct object is a good sign — the model located the relevant region before generating that token. Example: when generating "dog", low entropy + bright spot on the dog.
- **Diffuse attention** may indicate the model is uncertain about where to look, relying more on language priors than visual evidence. This often happens on abstract words like "because", "the", "a", or when the answer is not visually grounded.
- The maximum possible entropy for 49 regions (uniform distribution) is **ln(49) ≈ 3.89**. Any value near this means the attention is essentially uninformative.

The **entropy bar chart** in the Attention tab shows per-token entropy across the full answer sequence, making it easy to see which words were visually grounded and which were not.

---

## Example Questions

The models were trained on VQA v2.0 and generate free-form answers. These question styles work well:

### Object & Scene Recognition
- `What is in the image?`
- `What animal is in the picture?`
- `What type of vehicle is shown?`
- `What sport is being played?`
- `What room is this?`
- `What is on the table?`

### Color & Appearance
- `What color is the car?`
- `What color is the sky?`
- `What is the person wearing?`
- `How many people are in the image?`
- `Is it daytime or nighttime?`

### Location & Spatial
- `Where is the dog?`
- `What is in the background?`
- `What is on the left side of the image?`
- `Is the person indoors or outdoors?`

### Action & Activity
- `What is the person doing?`
- `What is the man holding?`
- `What is the child eating?`
- `Is the animal standing or sitting?`

### Yes / No (the model generates a sentence, not just "yes/no")
- `Is there a person in the image?`
- `Is the food cooked?`
- `Are there any trees?`
- `Is the image taken outdoors?`

### Descriptive (watch the confidence drop — these are harder)
- `What time of day is it?`
- `Why is the person smiling?`
- `How does the weather look?`
- `What is happening in this image?`

---

## API Reference

The backend exposes a simple REST API if you want to integrate or script queries.

### `GET /api/status`
Returns model load status and device info.

```json
{
  "device": "cuda",
  "device_name": "NVIDIA GeForce RTX 5070 Ti",
  "load_status": { "A": "loaded", "B": "loaded", "C": "loaded", "D": "loaded" },
  "vocab_loaded": true
}
```

### `GET /api/samples`
Returns 24 random COCO val2014 thumbnails as base64 JPEGs.

### `GET /api/refresh_samples`
Clears the sample cache and returns 24 new random thumbnails.

### `POST /api/infer`

**Request body:**
```json
{
  "image_b64": "data:image/jpeg;base64,/9j/...",
  "question": "What is in the image?",
  "models": ["A", "B", "C", "D"],
  "decode_mode": "both",
  "beam_width": 3
}
```

**Response (per model):**
```json
{
  "results": {
    "D": {
      "greedy_answer": "dog because a dog is sitting on a couch",
      "beam_answer": "dog sitting on a couch near a window",
      "tokens": ["dog", "because", "a", "dog", "is", "sitting", "..."],
      "token_probs": [0.9341, 0.4812, 0.8203, "..."],
      "avg_confidence": 0.7124,
      "top5_last": [
        {"word": "window", "prob": 0.43},
        {"word": "sofa", "prob": 0.21},
        "..."
      ],
      "has_attention": true,
      "heatmaps": ["<base64 PNG per token>", "..."],
      "avg_heatmap": "<base64 PNG>",
      "attn_weights": [[49 floats per token], "..."],
      "attn_entropy": [1.23, 3.45, 2.11, "..."],
      "meta": {"name": "Model D", "desc": "ResNet101 · Dual Attention", "color": "#8b5cf6"},
      "inference_ms": 49
    }
  },
  "display_image": "<base64 JPEG 224×224>"
}
```

**Script example:**
```python
import requests, base64

with open('my_image.jpg', 'rb') as f:
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()

r = requests.post('http://localhost:5000/api/infer', json={
    'image_b64': b64,
    'question': 'What is the dog doing?',
    'models': ['D'],
    'decode_mode': 'both',
    'beam_width': 3
})
result = r.json()['results']['D']
print('Greedy:', result['greedy_answer'])
print('Beam:  ', result['beam_answer'])
print('Avg confidence:', result['avg_confidence'])
```

---

## Requirements

```
torch >= 2.0
torchvision
flask
Pillow
matplotlib
numpy
```

The app auto-detects CUDA (RTX 5070 Ti or any Ampere+ GPU). Falls back to CPU if no GPU is available (inference will be slow, ~5–10× slower).

# MODEL G — DATA PHASE IMPLEMENTATION GUIDE
## Hướng dẫn chi tiết từng bước cho Data Pipeline

**Mục tiêu**: Từ raw data hiện tại → 219K filtered, multi-source training pool sẵn sàng cho Model G.

**Thời gian ước tính**: ~4-6 giờ (không tính download time)

**Yêu cầu**: Python 3.8+, ~5GB disk space cho processed data

---

## MỤC LỤC

1. [Kiểm kê hiện trạng](#1-kiểm-kê-hiện-trạng)
2. [Download file còn thiếu](#2-download-file-còn-thiếu)
3. [Sắp xếp thư mục](#3-sắp-xếp-thư-mục)
4. [Cài đặt dependencies](#4-cài-đặt-dependencies)
5. [Bước 1: Khám phá format data](#5-bước-1-khám-phá-format-data)
6. [Bước 2: Preprocess VQA-X](#6-bước-2-preprocess-vqa-x)
7. [Bước 3: Preprocess A-OKVQA](#7-bước-3-preprocess-a-okvqa)
8. [Bước 4: Preprocess VQA-E](#8-bước-4-preprocess-vqa-e)
9. [Bước 5: Merge tất cả sources](#9-bước-5-merge-tất-cả-sources)
10. [Bước 6: Filter chất lượng](#10-bước-6-filter-chất-lượng)
11. [Bước 7: Rebuild vocabulary](#11-bước-7-rebuild-vocabulary)
12. [Bước 8: Verify & sanity check](#12-bước-8-verify--sanity-check)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. KIỂM KÊ HIỆN TRẠNG

### Bạn đã có

```
✅ data/raw/train2014/              ~82K COCO images
✅ data/raw/val2014/                ~40K COCO images  
✅ data/raw/test2015/               COCO test images
✅ data/raw/vqa_data_json/          VQA v2.0 question + annotation JSONs
✅ data/vqa_e/VQA-E_train_set.json  ~210K VQA-E train samples
✅ data/vqa_e/VQA-E_val_set.json    ~88K VQA-E val samples
✅ A-OKVQA files                     aokvqa_v1p0_train.json, val, test + vocab CSVs
✅ VQA-X files                       train_x.json, val_x.json, test_x.json
```

### Còn thiếu

```
❌ instances_train2014.json          COCO object annotations (cần cho filter Stage 3)
❌ instances_val2014.json            COCO object annotations cho val images
```

---

## 2. DOWNLOAD FILE CÒN THIẾU

### 2.1 COCO Instances Annotations (BẮT BUỘC)

File `instances_train2014.json` chứa bounding boxes + category labels cho 80 loại
object trên mỗi COCO image. Đây là **ground truth** — chính xác hơn cả Faster R-CNN
detection. Cần cho filter visual grounding (Stage 3).

**Download:**

```bash
# Cách 1: Download trực tiếp (~250MB zip, giải nén ~18MB mỗi file)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

# Kết quả: thư mục annotations/ chứa:
#   instances_train2014.json    ← CẦN FILE NÀY
#   instances_val2014.json      ← VÀ FILE NÀY
#   captions_train2014.json     (không cần)
#   captions_val2014.json       (không cần)
#   person_keypoints_*.json     (không cần)
```

**Cách 2: Nếu wget không được**

Truy cập: https://cocodataset.org/#download

Trong mục "2014 Train/Val annotations" → download file "2014 Train/Val annotations [241MB]"

**Đặt file vào:**

```bash
# Copy 2 file instances vào thư mục data/raw/
cp annotations/instances_train2014.json data/raw/
cp annotations/instances_val2014.json data/raw/
```

### 2.2 Verify download

```bash
# Kiểm tra file tồn tại và size hợp lý
ls -lh data/raw/instances_train2014.json
# Expected: ~17-18MB

ls -lh data/raw/instances_val2014.json  
# Expected: ~7-8MB

# Quick test: đọc được JSON không
python -c "
import json
with open('data/raw/instances_train2014.json') as f:
    d = json.load(f)
print(f'Categories: {len(d[\"categories\"])}')   # Expected: 80
print(f'Annotations: {len(d[\"annotations\"])}') # Expected: ~600K+
print(f'Images: {len(d[\"images\"])}')           # Expected: ~82K
"
```

---

## 3. SẮP XẾP THƯ MỤC

Trước khi chạy bất kỳ script nào, tổ chức data theo cấu trúc chuẩn:

```bash
# Tạo thư mục cho VQA-X
mkdir -p data/vqa_x/raw

# Tạo thư mục cho A-OKVQA  
mkdir -p data/aokvqa/raw

# Di chuyển VQA-X files vào đúng chỗ
# (điều chỉnh path tùy nơi bạn download)
mv /path/to/train_x.json data/vqa_x/raw/
mv /path/to/val_x.json data/vqa_x/raw/
mv /path/to/test_x.json data/vqa_x/raw/

# Di chuyển A-OKVQA files vào đúng chỗ
mv /path/to/aokvqa_v1p0_train.json data/aokvqa/raw/
mv /path/to/aokvqa_v1p0_val.json data/aokvqa/raw/
mv /path/to/aokvqa_v1p0_test.json data/aokvqa/raw/
mv /path/to/large_vocab_train.csv data/aokvqa/raw/       # optional
mv /path/to/specialized_vocab_train.csv data/aokvqa/raw/  # optional
```

**Kiểm tra cấu trúc sau khi sắp xếp:**

```bash
# Chạy lệnh này để verify
find data/ -name "*.json" -o -name "*.csv" | head -20

# Expected output (thứ tự có thể khác):
# data/raw/instances_train2014.json
# data/raw/instances_val2014.json
# data/raw/vqa_data_json/v2_OpenEnded_mscoco_train2014_questions.json
# data/raw/vqa_data_json/v2_mscoco_train2014_annotations.json
# data/vqa_e/VQA-E_train_set.json
# data/vqa_e/VQA-E_val_set.json
# data/vqa_x/raw/train_x.json
# data/vqa_x/raw/val_x.json
# data/aokvqa/raw/aokvqa_v1p0_train.json
# data/aokvqa/raw/aokvqa_v1p0_val.json
```

---

## 4. CÀI ĐẶT DEPENDENCIES

```bash
# spaCy: cần cho noun extraction trong filter Stage 3
pip install spacy
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
```

Không cần GPU, không cần thêm package nào khác.

---

## 5. BƯỚC 1: KHÁM PHÁ FORMAT DATA

**MỤC ĐÍCH**: Xem chính xác format từng file trước khi viết preprocessing script. Đây là bước quan trọng nhất — nếu format khác expected thì script sẽ crash.

**Chạy script sau:**

```bash
python -c "
import json, os

def peek(path, n=2):
    '''Print first n entries of a JSON file'''
    if not os.path.exists(path):
        print(f'  ❌ FILE NOT FOUND: {path}')
        return
    with open(path) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        print(f'  Type: list, Length: {len(data)}')
        for i, item in enumerate(data[:n]):
            print(f'  [{i}] keys: {list(item.keys()) if isinstance(item, dict) else type(item)}')
            print(f'      {json.dumps(item, indent=6, ensure_ascii=False)[:500]}')
    elif isinstance(data, dict):
        print(f'  Type: dict, Top-level keys: {list(data.keys())[:10]}')
        # If dict of dicts (keyed by question_id)
        first_key = list(data.keys())[0]
        first_val = data[first_key]
        print(f'  First key: {first_key}')
        print(f'  First value: {json.dumps(first_val, indent=6, ensure_ascii=False)[:500]}')
    print()

print('='*60)
print('VQA-E (existing)')
print('='*60)
peek('data/vqa_e/VQA-E_train_set.json')

print('='*60)
print('VQA-X')  
print('='*60)
peek('data/vqa_x/raw/train_x.json')

print('='*60)
print('A-OKVQA')
print('='*60)
peek('data/aokvqa/raw/aokvqa_v1p0_train.json')

print('='*60)
print('COCO Instances (if downloaded)')
print('='*60)
path = 'data/raw/instances_train2014.json'
if os.path.exists(path):
    with open(path) as f:
        d = json.load(f)
    print(f'  Categories: {len(d[\"categories\"])}')
    print(f'  Sample category: {d[\"categories\"][0]}')
    print(f'  Annotations: {len(d[\"annotations\"])}')
    print(f'  Sample annotation: {json.dumps(d[\"annotations\"][0], indent=4)}')
else:
    print(f'  ❌ NOT FOUND: {path}')
    print(f'  → Download theo Section 2 của guide này')
"
```

**SAU KHI CHẠY**: Copy toàn bộ output và gửi lại cho tôi. Tôi sẽ dựa vào output chính xác này để viết preprocessing scripts không sai 1 byte.

**Tại sao bước này quan trọng?**

Các dataset VQA-X và A-OKVQA có nhiều phiên bản trên mạng, format có thể khác nhau:

- VQA-X: có version dùng `list` (mỗi entry là 1 dict), có version dùng `dict` (keyed by question_id). Key `explanation` có thể là string hoặc list. Key answer có thể là `multiple_choice_answer`, `answer`, hoặc `answers`.

- A-OKVQA: format khá chuẩn (list of dicts), nhưng `rationales` field có thể là list of strings hoặc nested structure.

Chạy script peek này mất < 5 giây, tiết kiệm hàng giờ debug sau này.

---

## 6. BƯỚC 2: PREPROCESS VQA-X

> **⚠️ CHỜ**: Bước này cần output từ Bước 1 để viết script chính xác. Sau khi bạn gửi output, tôi sẽ viết script `src/scripts/preprocess_vqa_x.py`.

### Mục tiêu

```
Input:  data/vqa_x/raw/train_x.json
Output: data/vqa_x/vqa_x_train_unified.json
        data/vqa_x/vqa_x_val_unified.json
```

### Logic (dự kiến, sẽ adjust theo format thực tế)

1. Đọc raw JSON
2. Cho mỗi entry:
   - Extract `image_id` → rename thành `img_id`
   - Extract `question`
   - Extract answer (từ field phù hợp)
   - Extract explanation (handle string/list)
   - Lowercase explanation
3. Verify `img_id` tồn tại trong COCO train2014 hoặc val2014:
   - Check file `COCO_train2014_{img_id:012d}.jpg` exists
   - Hoặc check `COCO_val2014_{img_id:012d}.jpg` exists
   - Skip nếu cả hai đều không có
4. Add `"source": "vqa_x"`
5. Save unified JSON + print statistics

### Expected output

```
VQA-X Preprocessing Report:
  Input:    29,459 samples
  Valid:    ~28,500 (images found in COCO 2014)
  Skipped:  ~959 (image not found)
  Saved:    data/vqa_x/vqa_x_train_unified.json
```

---

## 7. BƯỚC 3: PREPROCESS A-OKVQA

> **⚠️ CHỜ**: Cần output Bước 1.

### Mục tiêu

```
Input:  data/aokvqa/raw/aokvqa_v1p0_train.json
Output: data/aokvqa/aokvqa_train_unified.json
        data/aokvqa/aokvqa_val_unified.json
```

### Logic (dự kiến)

1. Đọc raw JSON
2. Cho mỗi entry:
   - Extract `image_id` → `img_id`
   - Extract `question`
   - Extract answer: mode of `direct_answers` list
   - Extract `rationales` (list of 3 strings)
   - Cho mỗi rationale:
     - Nếu rationale đã chứa answer → giữ nguyên
     - Nếu không → giữ nguyên (sẽ prepend ở training time)
   - Tạo entry với `"explanation": [rationale_1, rationale_2, rationale_3]`
3. Verify `img_id` tồn tại trong COCO 2014:
   - A-OKVQA dùng COCO 2017 image IDs nhưng hầu hết trùng COCO 2014
   - Check cả `train2014/` và `val2014/`
   - Skip nếu không tìm thấy (dự kiến mất ~2-5%)
4. Add `"source": "aokvqa"`
5. Save + statistics

### Đặc biệt: 3 rationales per question

A-OKVQA có 3 human-written rationales mỗi câu hỏi. Đây là **natural data augmentation**: mỗi epoch, training code sẽ random pick 1 trong 3 rationales (giống cách VQA-E code hiện tại pick random explanation). Không cần tách thành 3 entries riêng.

### Expected output

```
A-OKVQA Preprocessing Report:
  Input:      17,056 train questions (×3 rationales each = 51,168 explanation instances)
  Valid:      ~16,500 (images found in COCO 2014)
  Skipped:    ~556 (image not in COCO 2014)
  Saved:      data/aokvqa/aokvqa_train_unified.json
```

---

## 8. BƯỚC 4: PREPROCESS VQA-E

### Mục tiêu

VQA-E đã gần đúng format. Bước này chỉ thêm metadata `source` tag.

```
Input:  data/vqa_e/VQA-E_train_set.json
Output: data/vqa_e/vqa_e_train_unified.json
```

### Logic

1. Đọc VQA-E train JSON
2. Cho mỗi entry: add `"source": "vqa_e"`
3. Verify schema consistency (img_id, question, multiple_choice_answer, explanation)
4. Save + statistics

Script đơn giản nhất trong pipeline. ~20 dòng code.

---

## 9. BƯỚC 5: MERGE TẤT CẢ SOURCES

### Mục tiêu

```
Input:  data/vqa_e/vqa_e_train_unified.json
        data/vqa_x/vqa_x_train_unified.json
        data/aokvqa/aokvqa_train_unified.json

Output: data/processed/merged_train_raw.json
```

### Logic

1. Load tất cả unified files
2. Concatenate vào 1 list
3. Assign `length_bin` cho mỗi sample:
   - Tính word count của `"{answer} because {explanation[0]}"`
   - 1-5 words → `"short"`, 6-14 → `"medium"`, 15+ → `"long"`
4. Check duplicate: cùng img_id + cùng question → flag (không loại, để filter xử lý)
5. Print statistics report:
   - Count per source
   - Length distribution histogram (text-based)
   - Length bin distribution
   - Top 20 answers
6. Save merged file

### Expected output

```
Merge Report:
  VQA-E:    210,000 samples
  VQA-X:     28,500 samples  
  A-OKVQA:   16,500 samples
  TOTAL:    255,000 samples

  Length bins:
    short  (1-5 words):   12,300 (4.8%)
    medium (6-14 words):  89,200 (35.0%)
    long   (15+ words):  153,500 (60.2%)

  Saved: data/processed/merged_train_raw.json
```

---

## 10. BƯỚC 6: FILTER CHẤT LƯỢNG

### Mục tiêu

```
Input:  data/processed/merged_train_raw.json
        data/raw/instances_train2014.json
        data/raw/instances_val2014.json

Output: data/processed/merged_train_filtered.json  (samples pass all filters)
        data/processed/merged_train_scored.json     (all samples + quality_score)
        data/processed/filter_report.json           (statistics)
```

### Dependencies

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Logic: 5-Stage Quality Filter

**Stage 1 — Length Gate** (thuần Python)

```
Input:  explanation text (first element of explanation list)
Rule:   5 ≤ word_count(explanation) ≤ 35
Action: score -= 1 if fail
```

Quá ngắn (< 5 words): "because it is" — không có thông tin.
Quá dài (> 35 words): vượt LSTM generation capacity, sẽ bị truncate.

**Stage 2 — Copy-of-Question Detector** (thuần Python)

```
Input:  question Q, explanation E
Rule:   Jaccard(tokens(Q), tokens(E)) < 0.6
Action: score -= 1 if fail
```

Explanation chỉ rephrase question = zero information gain.

Ví dụ fail: Q: "Is the man riding a horse?" → E: "yes the man is riding a horse"
Jaccard = |{man,riding,horse}∩{man,riding,horse}| / |{man,riding,horse}∪{...}| ≈ 0.75 → FAIL

**Stage 3 — Visual Grounding** (cần spaCy + COCO instances)

```
Input:  explanation E, image objects from instances_train2014.json
Step 1: Extract content nouns from E using spaCy NER/POS
        nouns = {w for w in E if POS(w) in {NOUN, PROPN} and w not in stopwords}
Step 2: Load COCO objects for this image
        objects = category names from instances JSON
Step 3: Compute grounding ratio
        ratio = |nouns ∩ objects| / |nouns|
Rule:   IF |nouns| >= 2 AND ratio == 0.0 → FAIL
        (explanation mentions ≥2 concrete nouns but NONE appear in image)
Action: score -= 2 if fail (weighted heavier — hallucination is worst offense)
```

**Tại sao weight -2?** Hallucination là lỗi nghiêm trọng nhất cho Generative VQA. Model
learn sinh ra objects không có trong ảnh → catastrophic tại inference.

**Xử lý COCO instances format:**
```python
# instances_train2014.json structure:
# {
#   "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, ...],
#   "annotations": [{"image_id": 262148, "category_id": 1, "bbox": [...]}, ...],
#   "images": [{"id": 262148, "file_name": "COCO_train2014_000000262148.jpg"}, ...]
# }

# Build lookup: image_id → set of category names
cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
img_to_objects = defaultdict(set)
for ann in data["annotations"]:
    img_to_objects[ann["image_id"]].add(cat_id_to_name[ann["category_id"]])
# Result: {262148: {"person", "laptop", "chair"}, ...}
```

**Stage 4 — Answer Consistency** (thuần Python)

```
Input:  annotated answer A, full response text
Rule:   A (lowercased) appears in first 5 words of full response
Action: score -= 1 if fail
```

Catches: answer = "tennis" nhưng explanation nói "baseball because..."

**Stage 5 — Deduplication** (thuần Python)

```
Input:  all explanations for same img_id
Rule:   For each pair within same image:
        Jaccard(tokens(E1), tokens(E2)) < 0.85
Action: Keep first occurrence, mark duplicates score -= 1
```

### Scoring System

```
Max score: 5 (pass all stages)
Score breakdown:
  +1 per stage passed (stages 1,2,4,5)
  +2 for stage 3 passed (visual grounding worth double)

Decision:
  score 5:   GOLD   — full weight in training (1.0×)
  score 4:   GOOD   — slight downweight (0.8×)  
  score 3:   OK     — moderate downweight (0.5×)
  score ≤ 2: REJECT — excluded from training
```

### Expected output

```
Filter Report:
=============================================================
                    VQA-E     VQA-X    A-OKVQA    Total
-------------------------------------------------------------
Input               210000    28500    16500      255000
Stage 1 fail        16800     570      825        18195
Stage 2 fail        25200     855      165        26220  
Stage 3 fail        52500     1425     1320       55245
Stage 4 fail        6300      285      330        6915
Stage 5 fail        16800     285      0          17085
-------------------------------------------------------------
GOLD (score 5)      71400     23085    12705      107190
GOOD (score 4)      29400     1995     1155       32550
OK   (score 3)      12600     570      330        13500
REJECT (score ≤2)   96600     2850     2310       101760
-------------------------------------------------------------
Kept (score ≥3)     113400    25650    14190      153240
Pass rate           54.0%     90.0%    86.0%      60.1%
=============================================================

Saved: data/processed/merged_train_filtered.json (153,240 samples)
Saved: data/processed/merged_train_scored.json   (255,000 samples + scores)
Saved: data/processed/filter_report.json
```

*(Số liệu ước tính — actual sẽ khác)*

---

## 11. BƯỚC 7: REBUILD VOCABULARY

### Mục tiêu

```
Input:  data/processed/merged_train_filtered.json
        data/raw/vqa_data_json/v2_OpenEnded_mscoco_train2014_questions.json
        data/raw/vqa_data_json/v2_mscoco_train2014_annotations.json

Output: data/processed/vocab_questions.json  (rebuilt, larger)
        data/processed/vocab_answers.json    (rebuilt, larger)
```

### Logic

Script `1_build_vocab_v2.py` — mở rộng từ `1_build_vocab.py` hiện tại:

1. Load merged_train_filtered.json
2. Load VQA v2.0 questions + answers (giữ cho Phase 1 warm-up compatibility)
3. Build question vocab từ tất cả questions (VQA-E + VQA-X + A-OKVQA + VQA v2.0)
4. Build answer vocab từ tất cả `"{answer} because {explanation}"` strings
5. **Threshold = 5** (tăng từ 3, vì data pool lớn hơn → loại long-tail noise)
6. Save + print comparison với vocab cũ

### Expected output

```
Vocabulary Rebuild Report:
  Question vocab: 8,432 → ~9,200 tokens (+9.1%)
  Answer vocab:   8,648 → ~10,500 tokens (+21.4%)
  
  New tokens from VQA-X:    ~380
  New tokens from A-OKVQA:  ~1,100 (more diverse vocabulary due to knowledge reasoning)
  
  Saved: data/processed/vocab_questions.json
  Saved: data/processed/vocab_answers.json
```

**Lưu ý**: Vocab cũ sẽ được backup trước khi overwrite:
```bash
# Script sẽ tự động:
cp data/processed/vocab_questions.json data/processed/vocab_questions_backup.json
cp data/processed/vocab_answers.json data/processed/vocab_answers_backup.json
```

---

## 12. BƯỚC 8: VERIFY & SANITY CHECK

Sau khi hoàn thành tất cả, chạy verification script:

```bash
python -c "
import json

# 1. Check filtered data
with open('data/processed/merged_train_filtered.json') as f:
    data = json.load(f)
print(f'Total filtered samples: {len(data)}')

# 2. Source distribution
from collections import Counter
sources = Counter(d['source'] for d in data)
for src, cnt in sources.most_common():
    print(f'  {src}: {cnt} ({cnt/len(data)*100:.1f}%)')

# 3. Length distribution
lengths = [len(d['explanation'][0].split()) if d['explanation'] else 0 for d in data]
avg_len = sum(lengths) / len(lengths)
print(f'Avg explanation length: {avg_len:.1f} words')
print(f'  < 5 words:  {sum(1 for l in lengths if l < 5)} ({sum(1 for l in lengths if l < 5)/len(data)*100:.1f}%)')
print(f'  5-14 words:  {sum(1 for l in lengths if 5 <= l < 15)} ({sum(1 for l in lengths if 5 <= l < 15)/len(data)*100:.1f}%)')
print(f'  15-30 words: {sum(1 for l in lengths if 15 <= l <= 30)} ({sum(1 for l in lengths if 15 <= l <= 30)/len(data)*100:.1f}%)')
print(f'  > 30 words:  {sum(1 for l in lengths if l > 30)} ({sum(1 for l in lengths if l > 30)/len(data)*100:.1f}%)')

# 4. Check vocab
from src.vocab import Vocabulary
vq = Vocabulary(); vq.load('data/processed/vocab_questions.json')
va = Vocabulary(); va.load('data/processed/vocab_answers.json')
print(f'Question vocab: {len(vq)} tokens')
print(f'Answer vocab:   {len(va)} tokens')

# 5. Image coverage
img_ids = set(d['img_id'] for d in data)
import os
train_imgs = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir('data/raw/train2014') if f.endswith('.jpg'))
val_imgs = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir('data/raw/val2014') if f.endswith('.jpg'))
all_imgs = train_imgs | val_imgs
missing = img_ids - all_imgs
print(f'Unique images in data: {len(img_ids)}')
print(f'Missing images: {len(missing)}')
if missing:
    print(f'  First 5 missing: {list(missing)[:5]}')

print()
print('✅ Data pipeline verification complete!' if len(missing) == 0 else '⚠️  Some images missing — check above')
"
```

### Success criteria

```
✅ Total filtered samples: 140,000 - 180,000
✅ VQA-X: 25,000+ (≥ 85% of original)
✅ A-OKVQA: 13,000+ (≥ 80% of original)  
✅ Avg explanation length: 10-15 words
✅ 15-30 word range: ≥ 25% of samples
✅ Question vocab: 8,500 - 12,000 tokens
✅ Answer vocab: 9,000 - 13,000 tokens
✅ Missing images: 0 or < 100
```

---

## 13. TROUBLESHOOTING

### "File not found" errors

```
Q: instances_train2014.json not found
A: Download theo Section 2. File nằm trong zip annotations_trainval2014.zip

Q: Image not found cho A-OKVQA entries  
A: Bình thường. A-OKVQA dùng COCO 2017 IDs, ~2-5% có thể không match COCO 2014.
   Script sẽ skip và report count. Nếu > 10% missing → có thể bạn thiếu val2014 images.

Q: train_x.json format khác expected
A: Chạy peek script (Bước 1), gửi output lại. Script sẽ được adjust.
```

### spaCy errors

```
Q: "Can't find model 'en_core_web_sm'"
A: python -m spacy download en_core_web_sm

Q: spaCy quá chậm trên filter
A: Filter 255K samples mất ~10-15 phút với en_core_web_sm. Nếu cần nhanh hơn,
   có thể dùng nlp.pipe() với batch_size=1000 và n_process=4.
```

### Memory issues

```
Q: Merged JSON quá lớn, tràn RAM
A: merged_train_raw.json ~500MB-1GB. Cần ≥8GB RAM.
   Nếu thiếu: xử lý từng source riêng trong filter, merge filtered files cuối.
```

---

## TỔNG KẾT PIPELINE

```
Bước 1: Khám phá format     → gửi output cho tôi
Bước 2: preprocess_vqa_x.py → tôi viết sau khi có format
Bước 3: preprocess_aokvqa.py → tương tự
Bước 4: preprocess_vqae.py  → tương tự  
Bước 5: merge_all_sources.py
Bước 6: filter_quality.py   → script quan trọng nhất
Bước 7: 1_build_vocab_v2.py
Bước 8: verify

Tổng: 6 scripts Python, ~4-6 giờ thực hiện
Output: ~150-180K filtered samples, sẵn sàng cho Model G training
```

**HÀNH ĐỘNG TIẾP THEO**: Chạy Bước 1 (peek script), gửi output lại cho tôi. Tôi sẽ viết tất cả 6 scripts còn lại.

"""
Download script for Model G extra datasets:
  - A-OKVQA  : from S3 (allen AI)
  - VQA-X    : from Google Drive (with gdown)

Usage:
    cd /path/to/new_vqa
    pip install gdown tqdm
    python src/scripts/download_model_g_data.py
"""
import os
import sys
import subprocess
import urllib.request
import tarfile
import zipfile


BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')


def _check_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[INFO] gdown not found, installing ...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'])


def download_http(url, dest):
    """Download via urllib with progress."""
    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_gdrive(file_id, dest):
    """Download a single file from Google Drive via gdown."""
    _check_gdown()
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {os.path.basename(dest)} from Google Drive ...")
    try:
        gdown.download(url, dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  ✓ Saved to {dest}")
            return True
        else:
            print(f"  ✗ File empty or not found after download.")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


# ── A-OKVQA ───────────────────────────────────────────────────────────────────
def download_aokvqa():
    """Download & extract A-OKVQA v1.0 from Allen AI S3."""
    out_dir = os.path.join(DATA_DIR, 'aokvqa')
    os.makedirs(out_dir, exist_ok=True)

    # Check if already extracted
    expected_files = ['aokvqa_v1p0_train.json', 'aokvqa_v1p0_val.json']
    if all(os.path.exists(os.path.join(out_dir, f)) for f in expected_files):
        print("[A-OKVQA] Already downloaded. Skipping.")
        return

    tar_path = os.path.join(out_dir, 'aokvqa_v1p0.tar.gz')
    url = 'https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz'

    print("\n[A-OKVQA] Downloading from Allen AI S3 ...")
    if not os.path.exists(tar_path):
        ok = download_http(url, tar_path)
        if not ok:
            print("[A-OKVQA] FAILED. Please download manually from:")
            print("          https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz")
            return

    print(f"  Extracting {tar_path} ...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(out_dir)
        print("  ✓ Extracted to", out_dir)
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")


# ── VQA-X ─────────────────────────────────────────────────────────────────────
def download_vqa_x():
    """Download VQA-X JSON splits from Google Drive."""
    out_dir = os.path.join(DATA_DIR, 'vqa_x')
    os.makedirs(out_dir, exist_ok=True)

    # Google Drive file IDs for the 3 splits (Park et al. 2020)
    files = {
        'vqaX_train.json': '1gQNQL_8KMpaLQ-bVVBTL8L7y7LpOoYDO',
        'vqaX_val.json':   '1TaIROr1RrQX0rHifVmpq2dOiLJ7mlGIA',
        'vqaX_test.json':  '1xJXgCKcRJd0DAVFmVbdqjLqCxsm0Q2Jo',
    }

    print("\n[VQA-X] Downloading from Google Drive ...")
    for filename, file_id in files.items():
        dest = os.path.join(out_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  {filename} already exists. Skipping.")
            continue
        download_gdrive(file_id, dest)

    # Verify
    missing = [f for f in files if not os.path.exists(os.path.join(out_dir, f))
               or os.path.getsize(os.path.join(out_dir, f)) == 0]
    if missing:
        print(f"\n[VQA-X] WARNING: Some files are missing or empty: {missing}")
        print("  You can download VQA-X manually from:")
        print("  https://github.com/Seth-Park/RVT#download-data")
    else:
        print("[VQA-X] ✓ All splits downloaded successfully.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Base dir : {BASE_DIR}")
    print(f"Data dir : {DATA_DIR}")

    download_aokvqa()
    download_vqa_x()

    print("\nDone. Verify the following paths:")
    for subdir in ['aokvqa', 'vqa_x']:
        d = os.path.join(DATA_DIR, subdir)
        files = os.listdir(d) if os.path.isdir(d) else []
        print(f"  data/{subdir}/: {files or '[empty]'}")

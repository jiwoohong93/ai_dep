#!/usr/bin/env python3
"""
Disk-safe downloader for the ~4,376 chest X-ray images referenced by the two
expert-label CSVs, using the public HuggingFace mirror
(alkzar90/NIH-Chest-X-ray-dataset) instead of the rate-limited Kaggle API.

Per archive: stream-download images_XXX.zip -> extract only the NEEDED images
that aren't already on disk -> delete the zip. Peak disk ~= one archive (<=4GB)
plus the extracted PNGs (~2GB total). No Kaggle quota involved.

Output layout matches train.py / train_fair.py:
    data/datasets/nih-chest-xrays/data/versions/3/images_XXX/images/<file>.png
"""
import os
import csv
import zipfile
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_FILES = [
    os.path.join(BASE, "data/four_findings_expert_labels_test_labels.csv"),
    os.path.join(BASE, "data/four_findings_expert_labels_validation_labels.csv"),
]
VERSIONS_DIR = os.environ.get("NIH_ROOT", os.path.join(BASE, "data/datasets/nih-chest-xrays/data/versions/3"))
URL_TMPL = ("https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/"
            "resolve/main/data/images/images_{:03d}.zip")


def needed_images():
    names = set()
    for fn in CSV_FILES:
        with open(fn) as f:
            for row in csv.DictReader(f):
                names.add(row["Image Index"].strip())
    return names


def already_on_disk(versions_dir=None):
    versions_dir = versions_dir or VERSIONS_DIR
    present = set()
    for i in range(1, 13):
        d = os.path.join(versions_dir, f"images_{i:03d}", "images")
        if os.path.isdir(d):
            present.update(os.listdir(d))
    return present


def stream_download(url, dest, chunk=1 << 20):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with open(dest, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)
                got += len(c)
                if total:
                    print(f"\r    dl {got/1e9:.2f}/{total/1e9:.2f} GB", end="", flush=True)
        print()


def main(need=None, versions_dir=None):
    """need: set of image file names (default: the Four-Findings images);
    versions_dir: output root holding images_XXX/images/ (default: VERSIONS_DIR)."""
    need = set(need) if need is not None else needed_images()
    versions_dir = versions_dir or VERSIONS_DIR
    have = already_on_disk(versions_dir)
    missing = need - have
    print(f"[hf] need={len(need)} on_disk={len(need & have)} missing={len(missing)}")

    grand = 0
    for i in range(1, 13):
        if not missing:
            print("[hf] all needed images present; stopping early.")
            break
        url = URL_TMPL.format(i)
        zpath = os.path.join(versions_dir, f"_tmp_images_{i:03d}.zip")
        out_dir = os.path.join(versions_dir, f"images_{i:03d}", "images")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[hf] archive {i:03d}/012 -> {url.split('/')[-1]}  (missing so far {len(missing)})")
        try:
            stream_download(url, zpath)
            extracted = 0
            with zipfile.ZipFile(zpath) as z:
                for member in z.namelist():
                    base = os.path.basename(member)
                    if base in missing:
                        with z.open(member) as src, open(os.path.join(out_dir, base), "wb") as dst:
                            dst.write(src.read())
                        missing.discard(base)
                        extracted += 1
            grand += extracted
            print(f"[hf] archive {i:03d}: extracted {extracted} (remaining missing {len(missing)})")
        except Exception as e:
            print(f"[hf] archive {i:03d} FAILED: {repr(e)[:200]}")
        finally:
            if os.path.exists(zpath):
                os.remove(zpath)

    have = already_on_disk(versions_dir)
    print(f"[hf] DONE. extracted_this_run={grand} total_needed_on_disk={len(need & have)}/{len(need)} "
          f"still_missing={len(need - have)}")


if __name__ == "__main__":
    main()

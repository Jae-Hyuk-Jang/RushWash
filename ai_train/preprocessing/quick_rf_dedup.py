# quick_rf_dedup.py -----------------------------------------------------------
import argparse, re, shutil
from pathlib import Path
from tqdm import tqdm

RF_SUFFIX  = re.compile(r'\.rf\.[0-9a-f]{32}$', re.I)           # ← 핵심!
AUG_SUFFIX = re.compile(r'(_aug\d+|_flip|_rot\d+|_crop\d+)$', re.I)

def label_from_image(img_path: Path) -> Path:
    split = img_path.parents[1].name            # train / val / test
    lbl_dir = img_path.parents[2] / split / "labels"
    return lbl_dir / (img_path.stem + ".txt")

def move_pair(img_path: Path, trash_dir: Path):
    trash_dir.mkdir(exist_ok=True)
    shutil.move(img_path, trash_dir / img_path.name)
    lbl = label_from_image(img_path)
    if lbl.exists():
        shutil.move(lbl, trash_dir / lbl.name)

def main(dataset_root, trash="duplicates_rf", dryrun=False):
    ds = Path(dataset_root)
    kept_base = set()
    trash_dir = ds / trash

    for split in ["train", "val", "valid", "test"]:
        img_dir = ds / split / "images"
        if not img_dir.exists(): continue

        for img in tqdm(img_dir.glob("*.[jp][pn]g"), desc=split):
            stem = img.stem
            # 1️⃣ rf 해시 제거
            stem1 = RF_SUFFIX.sub("", stem)
            # 2️⃣ 일반 증강 suffix 제거 (선택)
            base  = AUG_SUFFIX.sub("", stem1)

            if base in kept_base:
                if not dryrun:
                    move_pair(img, trash_dir)
            else:
                kept_base.add(base)

    print(f"[Done] name-based rf-dedup finished. Duplicates → {trash_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--trash", default="duplicates_rf")
    ap.add_argument("--dryrun", action="store_true")
    main(**vars(ap.parse_args()))

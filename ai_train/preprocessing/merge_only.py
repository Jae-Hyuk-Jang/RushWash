#!/usr/bin/env python
# merge_only.py ──────────────────────────────────────────────
# 모든 입력 YOLO 데이터셋(train/val/test 구조)을 한 폴더(all/images·labels)
# 로만 합칩니다. split은 신경 쓰지 않습니다.
# ────────────────────────────────────────────────────────────
import argparse, shutil, os, yaml
from pathlib import Path
from tqdm import tqdm

def copy_pair(src_img: Path, dst_dir: Path, hardlink=False):
    dst_img = dst_dir / (src_img.stem + src_img.suffix)
    if hardlink:
        os.link(src_img, dst_img)
    else:
        shutil.copy2(src_img, dst_img)

    src_lbl = src_img.with_suffix(".txt").parent.parent / "labels" / (src_img.stem + ".txt")
    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_dir.parent / "labels" / src_lbl.name)

def main(roots, out_root, master_yaml=None, hardlink=False):
    out = Path(out_root)
    (out / "all" / "images").mkdir(parents=True, exist_ok=True)
    (out / "all" / "labels").mkdir(parents=True, exist_ok=True)

    for r in roots:
        r = Path(r)
        for split in ["train", "val", "valid", "test"]:
            img_dir = r / split / "images"
            if not img_dir.exists(): continue
            for img in tqdm(img_dir.glob("*.[jp][pn]g"), desc=f"{r.name}/{split}"):
                copy_pair(img, out / "all" / "images", hardlink)

    # data.yaml(임시) ─ split 안 쓰고 all/images 만 가리킴
    if master_yaml:
        names = yaml.safe_load(open(master_yaml, encoding="utf-8"))["names"]
    else:
        # 첫 roots 중 data.yaml 에서 names 로드
        names = yaml.safe_load(open(Path(roots[0])/"data.yaml", encoding="utf-8"))["names"]

    yaml.safe_dump({
        "path": str(out.resolve()),
        "train": "all/images",
        "nc": len(names),
        "names": names
    }, open(out/"data.yaml", "w", encoding="utf-8"), allow_unicode=True)
    print("✅ merge-only 완료 →", out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, help="remap 폴더들")
    p.add_argument("--out_root", required=True, help="하나로 합칠 폴더")
    p.add_argument("--master_yaml", help="master.yaml (없으면 첫 roots 의 data.yaml 사용)")
    p.add_argument("--hardlink", action="store_true")
    args = p.parse_args()
    main(args.roots, args.out_root, args.master_yaml, args.hardlink)

#!/usr/bin/env python
"""
merge_and_split.py
──────────────────────────────────────────────────────
  • 여러 YOLO 데이터셋(각각 train/val/test 구조)을 통합
  • 클래스 비율을 유지(Stratified)하며 train/val/test 재구성
  • master.yaml or 기존 data.yaml(names) 를 기준으로 class id 유지
──────────────────────────────────────────────────────
usage:
python merge_and_split.py \
  --roots 233images_laundrySymbol_remap 264images_laundrySymbol_remap \
          539images_laundrySymbol_remap 3251images_laundrySymbol_remap \
  --out_root combined_dataset \
  --val_ratio 0.1 --test_ratio 0.1 \
  --master_yaml C:/LAB/python/master.yaml \
  --hardlink
"""
import argparse, os, random, shutil, yaml, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_names(master_yaml: Path | None, sample_data_yaml: Path | None):
    """load class names list from master.yaml (우선) or sample data.yaml"""
    if master_yaml and Path(master_yaml).exists():
        with open(master_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["names"]
    if sample_data_yaml and sample_data_yaml.exists():
        with open(sample_data_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["names"]
    sys.exit("❌ names list를 찾을 수 없습니다.  --master_yaml 옵션을 지정하세요.")

def stratified_split(paths, class_ids, val_ratio, test_ratio):
    """paths·class_ids 길이 동일. 클래스별 비율 유지하며 split"""
    by_cls = defaultdict(list)
    for p, cid in zip(paths, class_ids):
        by_cls[cid].append(p)

    train, val, test = [], [], []
    rnd = random.Random(42)
    for cid, items in by_cls.items():
        rnd.shuffle(items)
        n = len(items)
        n_val  = round(n * val_ratio)
        n_test = round(n * test_ratio)
        val  += items[:n_val]
        test += items[n_val:n_val+n_test]
        train += items[n_val+n_test:]
    rnd.shuffle(train), rnd.shuffle(val), rnd.shuffle(test)
    return train, val, test

def copy_pair(src_img: Path, dst_img: Path, hardlink=False):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(src_img, dst_img)
    else:
        shutil.copy2(src_img, dst_img)
    # label
    src_lbl = src_img.with_suffix(".txt").parent.parent/ "labels" / (src_img.stem + ".txt")
    dst_lbl = dst_img.parent.parent/ "labels" / (dst_img.stem + ".txt")
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_lbl, dst_lbl)

def main(args):
    roots = [Path(r) for r in args.roots]
    out_root = Path(args.out_root)
    if out_root.exists():
        print(f"⚠️  '{out_root}' 이미 존재 → 덮어쓰지 않으려면 폴더명을 바꾸세요.")
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- names 로드 --------------------------------------------------------
    sample_data_yaml = roots[0]/"data.yaml"
    names = load_names(Path(args.master_yaml) if args.master_yaml else None,
                       sample_data_yaml)
    name2id = {n:i for i,n in enumerate(names)}

    # ----- 모든 이미지·라벨 수집 --------------------------------------------
    img_list, cls_ids = [], []
    for root in roots:
        for split in ["train","val","valid","test","all"]:
            img_dir = root/split/"images"
            lbl_dir = root/split/"labels"
            if not img_dir.exists(): continue
            for img in img_dir.glob("*.[jp][pn]g"):
                lbl = lbl_dir/(img.stem + ".txt")
                if not lbl.exists(): continue
                first_id = int(lbl.read_text().split()[0])
                cls_ids.append(first_id)
                img_list.append(img)

    print(f"📦 총 {len(img_list):,} 장 수집 완료.")

    train, val, test = stratified_split(
        img_list, cls_ids, args.val_ratio, args.test_ratio)

    splits = {"train": train, "val": val, "test": test}
    for split, items in splits.items():
        for img in tqdm(items, desc=f"copy {split}", unit="img"):
            dst_img = out_root/split/"images"/img.name
            copy_pair(img, dst_img, hardlink=args.hardlink)

    # ----- data.yaml 작성 ----------------------------------------------------
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc": len(names),
        "names": names
    }
    yaml.safe_dump(data_yaml, open(out_root/"data.yaml","w",encoding="utf-8"),
                   allow_unicode=True, sort_keys=False)
    print("✅ merge & split 완료 →", out_root/"data.yaml")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True,
                   help="통합할 remap 데이터셋 폴더들")
    p.add_argument("--out_root", required=True,
                   help="출력을 저장할 새 폴더")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--master_yaml", help="master.yaml 경로 (없으면 첫 데이터셋의 data.yaml에서 names 로드)")
    p.add_argument("--hardlink", action="store_true",
                   help="이미지 복사를 NTFS 하드링크로 대체(같은 드라이브에서만 가능)")
    args = p.parse_args()
    main(args)

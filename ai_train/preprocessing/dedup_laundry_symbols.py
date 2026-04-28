# dedup_laundry_symbols.py
# -----------------------------------------------------------------------------
# 1) 파일명 규칙 기반 증강본 제거
# 2) pHash 유사도 기반 이미지 중복 제거
#    이미지가 이동될 때 같은 stem의 .txt 라벨도 함께 이동
# -----------------------------------------------------------------------------
import argparse, os, shutil, re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import imagehash

# ---- ① 증강 suffix 패턴 정의 -------------------------------------------------
AUG_REGEX = re.compile(r'(_aug\d+|_flip|_rot\d+|_crop\d+)$', re.I)

# ---- ② 이미지 경로 → 라벨 경로 변환 -----------------------------------------
def label_from_image(img_path: Path) -> Path:
    """
    images/train/img001.jpg  →  labels/train/img001.txt
    """
    split = img_path.parents[1].name          # train / val / test
    lbl_dir = img_path.parents[2] / split / "labels"
    return lbl_dir / (img_path.stem + ".txt")

# ---- ③ 이동 함수 : 이미지 + 라벨 동시 이동 -----------------------------------
def move_to_trash(img_path: Path, trash_dir: Path):
    trash_dir.mkdir(exist_ok=True)
    # 이미지
    shutil.move(str(img_path), str(trash_dir / img_path.name))
    # 라벨
    lbl_path = label_from_image(img_path)
    if lbl_path.exists():
        trash_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(lbl_path), str(trash_dir / lbl_path.name))

# ---- ④ 메인 로직 ------------------------------------------------------------
def main(dataset_root, threshold=6, trash="duplicates", dryrun=False):
    ds = Path(dataset_root)
    trash_dir = ds / trash
    hash_map = {}                # perceptual hash → 기준 이미지
    base_seen = set()            # 증강 원본(base) 중복 체크

    for split in ["train", "val", "valid", "test"]:
        img_dir = ds / split / "images"
        if not img_dir.exists():
            continue

        for img_path in tqdm(list(img_dir.glob("*.[jp][pn]g")), desc=split):
            stem = img_path.stem
            base = AUG_REGEX.sub("", stem)    # suffix 제거

            # --- 증강본 중복 ----------------
            if base in base_seen:
                if not dryrun:
                    move_to_trash(img_path, trash_dir)
                continue
            base_seen.add(base)

            # --- pHash 중복 -----------------
            ph = imagehash.phash(Image.open(img_path))
            dup = False
            for ref_h in hash_map:
                if abs(ph - ref_h) <= threshold:
                    dup = True
                    break
            if dup:
                if not dryrun:
                    move_to_trash(img_path, trash_dir)
            else:
                hash_map[ph] = img_path

    # 통계
    print(f"[Done] duplicates saved to: {trash_dir}")

# ---- ⑤ CLI ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="train/val/test 상위 폴더 경로")
    ap.add_argument("--threshold", type=int, default=6,
                    help="pHash 해밍거리 허용치 (0=완전 동일, 6~8 권장)")
    ap.add_argument("--trash", default="duplicates",
                    help="이동 대상 저장 폴더명")
    ap.add_argument("--dryrun", action="store_true",
                    help="파일을 실제로 이동하지 않고 통계만 출력")
    args = ap.parse_args()
    main(**vars(args))

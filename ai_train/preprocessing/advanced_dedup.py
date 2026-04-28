# advanced_dedup.py -----------------------------------------------------------
# 1) 규칙적 파일명 suffix 기반 증강본 제거
# 2) 회전 보정 multi-pHash(0,90,180,270°) 해밍거리 ≤ hash_thr
# 3) CLIP 임베딩 코사인 유사도 ≥ clip_thr 로 최종 중복 판정
# ---------------------------------------------------------------------------
import argparse, os, re, shutil, itertools, faiss, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import imagehash, open_clip

# ---------- 설정 가능한 파라미터 -------------------------------------------
AUG_REGEX = re.compile(r'(_aug\d+|_flip|_rot\d+|_crop\d+)$', re.I)

# ---------- 유틸: 이미지→라벨 경로 -----------------------------------------
def label_from_image(img_path: Path) -> Path:
    split = img_path.parents[1].name                 # train / val / test
    lbl_dir = img_path.parents[2] / split / "labels"
    return lbl_dir / f"{img_path.stem}.txt"

# ---------- move 함수: 이미지+라벨 동시 이동 --------------------------------
def move_pair(img_path: Path, trash_dir: Path):
    trash_dir.mkdir(exist_ok=True)
    shutil.move(str(img_path), str(trash_dir / img_path.name))
    lbl = label_from_image(img_path)
    if lbl.exists():
        shutil.move(str(lbl), str(trash_dir / lbl.name))

# ---------- Multi-pHash (0/90/180/270) --------------------------------------
def multi_phash(img: Image.Image):
    return [imagehash.phash(img.rotate(r, expand=True)) for r in (0, 90, 180, 270)]

# ---------------------------------------------------------------------------
def main(dataset_root, hash_thr=6, clip_thr=0.94, trash="duplicates",
         dryrun=False, use_gpu=True):
    ds = Path(dataset_root)
    trash_dir = ds / trash

    # --- CLIP 모델 준비 ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.to(device).eval()

    def clip_vec(path: Path):
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
        return torch.nn.functional.normalize(emb, dim=-1)[0].cpu().numpy()

    # --- 1차 패스: 증강 suffix + multi-pHash 중복 ---------------------------
    uniq_paths = []          # pHash & suffix 필터 통과한 이미지
    hash_bank = []           # [(hashes list, Path)]

    for split in ["train", "val", "valid", "test"]:
        img_dir = ds / split / "images"
        if not img_dir.exists(): continue

        base_seen = set()    # 같은 prefix(base) 면 첫 장만 남김
        for img_path in tqdm(list(img_dir.glob("*.[jp][pn]g")), desc=f"{split}-pass1"):
            stem, base = img_path.stem, None
            if AUG_REGEX.search(stem):
                base = AUG_REGEX.sub("", stem)
            if base and base in base_seen:
                if not dryrun:
                    move_pair(img_path, trash_dir)
                continue
            if base: base_seen.add(base)

            img = Image.open(img_path)
            hashes = multi_phash(img)
            dup = False
            for h_list, _ in hash_bank:
                if min(abs(h - h0) for h in hashes for h0 in h_list) <= hash_thr:
                    dup = True
                    break
            if dup:
                if not dryrun:
                    move_pair(img_path, trash_dir)
            else:
                hash_bank.append((hashes, img_path))
                uniq_paths.append(img_path)

    # --- 2차 패스: CLIP 임베딩 중복(잘린·회전까지) ---------------------------
    if len(uniq_paths) <= 1:
        print("[INFO] No images left for CLIP phase.")
        return

    vec_dim = 512
    index = faiss.IndexFlatIP(vec_dim)
    vecs = []
    for p in tqdm(uniq_paths, desc="CLIP-emb"):
        vecs.append(clip_vec(p))
    vecs = torch.stack([torch.tensor(v) for v in vecs]).numpy()
    index.add(vecs)

    to_remove = set()
    for i, vec in enumerate(tqdm(vecs, desc="CLIP-dedup")):
        if i in to_remove: continue
        _, idxs = index.search(vec.reshape(1, -1), 10)   # top-10 후보
        for j in idxs[0]:
            if j == i or j in to_remove: continue
            sim = (vec @ vecs[j]).item()
            if sim >= clip_thr:
                to_remove.add(j)
                if not dryrun:
                    move_pair(uniq_paths[j], trash_dir)

    print(f"[Done] {len(to_remove)} additional CLIP duplicates moved to {trash_dir}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="train/val/test 상위 경로")
    ap.add_argument("--hash_thr", type=int, default=6,
                    help="multi-pHash 해밍거리 임계값 (0~64)")
    ap.add_argument("--clip_thr", type=float, default=0.94,
                    help="CLIP 코사인 유사도 임계값 (0~1)")
    ap.add_argument("--trash", default="duplicates",
                    help="이동 대상 폴더명")
    ap.add_argument("--dryrun", action="store_true",
                    help="파일 이동 없이 통계만 출력")
    ap.add_argument("--cpu", action="store_true",
                    help="강제로 CPU 실행 (GPU 없으면 자동 CPU)")
    args = ap.parse_args()
    main(args.dataset_root, args.hash_thr, args.clip_thr,
         args.trash, args.dryrun, not args.cpu)

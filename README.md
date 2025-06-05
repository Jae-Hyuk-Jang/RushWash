# 🧺 RushWash: AI-Based Laundry Assistant  
> YOLOv8 + CNN + OCR + LLM 기반 얼룩 & 세탁 기호 분석 자동화 시스템  
> Graduation Comprehensive Project (2025)

---

## 📌 프로젝트 개요
**RushWash**는 의류 이미지에서 얼룩과 세탁 기호를 자동으로 인식하고,  
사용자에게 **가장 적절한 세탁 방법**을 **자연어로 추천**하는 AI 서비스입니다.

- **얼룩 검출** : YOLOv8 (Object Detection)  
- **얼룩 분류** : ConvNeXt (Image Classification)  
- **세탁 기호 인식** : YOLO (Object Detection)  
- **세탁법 제안** : LLM (OpenAI Prompt)  
- **엔드-투-엔드 파이프라인** : Python + Spring Boot + React + MariaDB  

---

## 🧠 주요 기능

| 기능명 | 설명 |
|---|---|
| 🟥 얼룩 탐지 | YOLOv8로 의류 내 얼룩 Bounding Box 검출 |
| 🟦 얼룩 종류 분류 | 검출된 영역 crop → ConvNeXt로 분류 |
| 🟨 세탁 기호 인식 | 세탁 라벨을 YOLO로 검출 |
| 🟩 세탁 가이드 생성 | 얼룩 + 섬유 정보 기반 LLM 문장 생성 |
| 🛠️ MLOps 파이프라인 | MariaDB 연동 → 학습/평가 → JSON 결과 자동 저장 |

---

## 🏗️ 시스템 아키텍처
```plaintext
[Input Image]
     │
     ▼
  [YOLOv8]
     │
     ├──→ 얼룩 검출 ──→ [CNN] 얼룩 분류
     │
     └──→ 세탁 기호 인식 (YOLO)
     │
     ▼
   [LLM] 얼룩 + 섬유 기반 세탁 가이드 문장 생성
     ▼
[Frontend (React)] 사용자에게 결과 시각화
````

---

## ⚙️ 기술 스택

| 분류                   | 사용 기술                                            |
| -------------------- | ------------------------------------------------ |
| Object Detection     | **YOLOv8** (Ultralytics)                         |
| Image Classification | **ConvNeXt-Tiny** (timm)                         |
| OCR                  | **Tesseract** + Post-processing                  |
| 텍스트 생성               | **OpenAI Prompt** (LLM)                          |
| 백엔드                  | **Spring Boot**, **MariaDB**                     |
| 프론트엔드                | **React**, **Tailwind CSS**                      |
| ML Infra             | **Python**, **PyTorch**, **Ray Tune**, **WandB** |

---

## 📊 성능 요약

| 모델                     | Top-1 Acc              | Top-3 Acc | mAP        |
| ---------------------- | ---------------------- | --------- | ---------- |
| YOLOv8s (imgsz = 1920) | 69.16 %                | 79.44 %   | **88 % +** |
| ConvNeXt (얼룩 분류)       | 71.20 %                | 84.30 %   | —          |
| YOLO (세탁 기호)           | —                      | —         | **≈ 91 %** |
| LLM 세탁 가이드             | 자연스러운 문장 생성 (정량 평가 불가) |           |            |

---

## 🧪 사용 방법

```bash
# 1) 전체 파이프라인 실행
python pipe_final.py \
  --db-host localhost \
  --db-port <PORT> \
  --db-user <USER> \
  --db-password <PASSWORD> \
  --db-name rushwash

# 2) 개별 추론 (얼룩 이미지 예시)
python ai/infer_stain.py --image /path/to/image.jpg
```

```bash
# 3) 프론트엔드 실행
cd frontend
npm install
npm run dev
```

---

## 📁 프로젝트 구조

```plaintext
RushWash/
├── ai/                # YOLO, CNN, OCR, LLM 모듈
├── data/              # 학습 데이터셋 · 설정
├── frontend/          # React UI
├── backend/           # Spring Boot 서버
├── database/          # MariaDB 스키마
├── pipe_final.py      # 전체 파이프라인 스크립트
└── README.md
```

---

## 🏆 수상 및 발표

* 🥇 **한국인공지능융합기술학회 우수 발표 논문상** (2025)

---

## 👤 개발자

| 이름                      | 역할                            |
| ----------------------- | ----------------------------- |
| **장재혁 (Jang Jae-hyuk)** | 시스템 설계·구현 (AI, MLOps) |


---

## 📄 라이선스

이 프로젝트는 [MIT License](./LICENSE)에 따라 배포됩니다.

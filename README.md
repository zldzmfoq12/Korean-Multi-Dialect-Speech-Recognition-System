# 교통약자를 위한 다중 방언 음성인식

고령자 등 교통약자의 한국어 음성 명령을 인식하는 연구용 음성인식 시스템입니다. Whisper에 방언 임베딩을 입력하고, 이 임베딩으로 인코더 어텐션의 LoRA 가중치를 생성합니다. 학습과 추론 시 발화의 방언 정보를 입력받으며, 음성은 16 kHz로 변환해 처리합니다.

| 추론 ID | 학습 태그 | 방언 |
| --- | --- | --- |
| 0 | `se` | 표준어·수도권 |
| 1 | `jl` | 전라 |
| 2 | `gs` | 경상 |
| 3 | `cc` | 충청 |
| 4 | `gw` | 강원 |
| 5 | `jj` | 제주 |

## 설치

Python 3.10 이상을 사용합니다. 학습에는 BF16을 지원하는 CUDA GPU가 필요합니다. 처음 실행할 때 기본 Whisper 모델을 다운로드합니다.

```bash
python -m pip install -r requirements.txt
```

## 데이터 준비 및 학습

AI Hub 음성·전사 데이터는 승인받은 사용자가 AI Hub에서 직접 내려받아 로컬에서 준비합니다. 원본과 전처리 데이터, 파일 목록, 모델 체크포인트, 학습·평가 결과는 배포 범위에서 제외합니다.

학습과 검증용 목록을 각각 준비한 뒤 `config/config.py`의 `wav_scp`, `text_file`, `val_wav_scp`, `val_text_file`에 경로를 지정합니다. 기본 경로는 각각 `data/train/`과 `data/dev/` 아래에 있습니다. 두 목록은 UTF-8로 작성하고 발화 ID를 일치시킵니다.

| 파일 | 한 줄 형식 |
| --- | --- |
| `wav.scp` | `<발화_ID> <로컬_음성_파일_경로>` |
| `text` | `<발화_ID> <방언_태그> <전사문>` |

음성 파일 경로에 공백을 넣을 수 없으며, 학습 입력은 길이 0.1~30초의 음성입니다. 학습 중 검증 데이터의 문자 오류율(CER)을 계산하고 가장 낮은 CER을 기록한 모델을 `exp/best_model.pth`에 저장합니다.

```bash
python train_lora.py
```

여러 GPU를 사용하려면 예를 들어 `accelerate launch --multi_gpu --num_processes 2 train_lora.py`로 실행합니다.

## 추론

학습해서 얻은 체크포인트와 직접 준비한 음성 파일을 지정합니다. `--dialect`에 위 표의 방언 ID를 입력합니다. 아래 명령은 전라 방언을 지정합니다.

```bash
python inference.py --audio /path/to/audio.wav --dialect 1 --ckpt exp/best_model.pth
```

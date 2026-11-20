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

AI Hub 음성·전사 데이터는 승인받은 사용자가 AI Hub에서 직접 내려받아 로컬에서 준비합니다. 원본과 전처리 데이터, 파일 목록 및 학습·평가 결과는 배포 범위에서 제외합니다.

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

### 전처리

`prepare_data.py`는 음성과 전사·라벨을 발화 ID로 연결합니다. 라벨 JSON에 `transcription.sentences`가 있으면 구간별로 분할합니다. 이후 전사문 공백 정리, 방언 태그 부여, 너무 짧거나 긴 음성 및 무음 제거, 16 kHz·16-bit PCM 모노 변환을 수행합니다. 입력 목록은 각 줄에 `<발화_ID> <파일_경로>`를 적고, 전사 목록은 `<발화_ID> [방언_태그] <전사문>` 형식을 사용합니다. 학습·검증 세트를 먼저 나누고 각각 실행합니다. 두 목록을 출력할 폴더의 경로에는 공백을 넣지 않습니다.

```bash
python -m pip install soundfile scipy
python prepare_data.py --wav-scp local/train/wav.scp --text local/train/text \
  --label-scp local/train/label.scp --output-dir data/train
```

문장 분할 전사 필드의 기본값은 `dialect`이며 `--transcript-field`로 지정할 수 있습니다. `--label-scp`를 생략하면 각 WAV 옆에서 같은 이름의 JSON을 찾습니다.

선택적으로 Faster-Whisper의 평균 로그 확률을 이용해 신뢰도가 낮은 발화를 걸러냅니다. 출력은 음성 경로와 전사 목록이며, 학습 설정에서 필터링 결과를 지정할 수 있습니다.

```bash
python -m pip install faster-whisper
python filter_mismatch_whisper.py --wav-scp data/train/wav.scp --text data/train/text \
  --output-dir data/train_verified --device cuda --threshold 0.5
```

## 추론

학습해서 얻은 체크포인트와 직접 준비한 음성 파일을 지정합니다. `--dialect`에 위 표의 방언 ID를 입력합니다. 아래 명령은 전라 방언을 지정합니다.

```bash
python inference.py --audio /path/to/audio.wav --dialect 1 --ckpt exp/best_model.pth
```

## Docker CPU 추론

[릴리스](https://github.com/zldzmfoq12/Korean-Multi-Dialect-Speech-Recognition-System/releases/tag/v0.1.0)에서 `moe_whisper_ep5.pth`를 내려받고 추론할 `audio.wav`와 같은 폴더에 둡니다. 처음 추론할 때 기본 Whisper 모델과 토크나이저를 내려받습니다. Linux x86-64 이미지를 사용합니다.

```bash
docker run --rm -v "$PWD:/input:ro" \
  ghcr.io/zldzmfoq12/korean-multi-dialect-speech-recognition-system:v0.1.0 \
  --audio /input/audio.wav --dialect 1 --ckpt /input/moe_whisper_ep5.pth
```

import torch
from torch.utils.data import Dataset
import torchaudio

class KaldiMoEDataset(Dataset):
    def __init__(self, wav_scp, text_file, processor):
        """
        wav_scp: 오디오 경로 파일 (key path)
        text_file: 방언 텍스트 및 정보 파일 (key dialect dialect_transcript)
        processor: Whisper Processor
        """
        self.processor = processor
        self.resamplers = {}

        # 방언 ID 매핑
        self.dialect_map = {
            "se": 0, "jl": 1, "gs": 2, "cc": 3, "gw": 4, "jj": 5,
        }

        # -------------------------------------------------------
        # 1. 파일 파싱
        # -------------------------------------------------------
        wav_paths = self._parse_scp(wav_scp)
        transcripts, dialect_info = self._parse_text_and_dialect(text_file)

        # -------------------------------------------------------
        # 2. 데이터 병합 (wav.scp 기준 Intersection)
        # -------------------------------------------------------
        self.data = []
        print(f"🔄 Merging data...")

        for key, wav_path in wav_paths.items():
            if key in transcripts:
                self.data.append({
                    "key": key,
                    "wav_path": wav_path,
                    "text": transcripts[key],
                    "dialect": dialect_info.get(key, "0")
                })

        print(f"✅ Loaded {len(self.data)} samples.")

        # 3. 길이 정보 계산 (Smart Batching용)
        self.lengths = []
        for item in self.data:
            self.lengths.append(len(item['text']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 무한 루프 방지용 카운터
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            curr_idx = (idx + attempts) % len(self)
            item = self.data[curr_idx]

            wav_path = item['wav_path']
            transcript = item['text']
            raw_dialect = item['dialect']

            # 1. 텍스트 누락 체크
            if not transcript or len(transcript.strip()) == 0:
                attempts += 1
                continue

            # 2. 오디오 로드
            try:
                audio, sr = torchaudio.load(wav_path)
            except Exception as e:
                print(f"⚠️ Error loading {wav_path}: {e}")
                attempts += 1
                continue

            # 3. 0바이트 체크
            if audio.numel() == 0:
                attempts += 1
                continue

            # 4. 너무 짧은 오디오 (< 0.1초)
            if audio.shape[1] < sr * 0.1:
                attempts += 1
                continue

            # 5. 리샘플링 (16kHz)
            if sr != 16000:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio = self.resamplers[sr](audio)

            # ===============================================================
            # [유지] 30초 초과 시 스킵 (Skip > 30s)
            # ===============================================================
            MAX_SAMPLES = 480000 # 30초 * 16000Hz
            if audio.shape[1] > MAX_SAMPLES:
                # print(f"⚠️ Skip long audio (>30s): {wav_path}")
                attempts += 1
                continue

            # 유효한 데이터 처리 시작

            # Processor (Audio)
            input_features = self.processor(
                audio.mean(dim=0).numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features[0]

            # 방언 라벨 (Target)
            label_features = self.processor.tokenizer(
                transcript,
                max_length=448,
                truncation=True,
                return_attention_mask=False
            )
            labels = label_features.input_ids

            # 방언 ID
            dialect_id = self.dialect_map.get(str(raw_dialect), 0)

            return {
                "input_features": input_features,
                "labels": torch.tensor(labels, dtype=torch.long),
                "dialect_id": torch.tensor(dialect_id, dtype=torch.long)
            }

        raise RuntimeError(f"No usable audio found starting at dataset index {idx}.")

    def _parse_scp(self, path):
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data[parts[0]] = parts[1]
        return data

    def _parse_text_and_dialect(self, path):
        texts = {}
        dialects = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    key = parts[0]
                    dialects[key] = parts[1]
                    texts[key] = " ".join(parts[2:])
                elif len(parts) == 2:
                    key = parts[0]
                    texts[key] = parts[1]
                    dialects[key] = "0"
        return texts, dialects

class DataCollator:
    def __call__(self, batch):
        # None 필터링
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        input_features = [item['input_features'] for item in batch]
        labels = [item['labels'] for item in batch]
        dialect_ids = [item['dialect_id'] for item in batch]

        input_features = torch.stack(input_features)
        dialect_ids = torch.stack(dialect_ids)

        # Labels Padding (-100 for Loss)
        max_label_len = max([len(l) for l in labels])
        padded_labels = torch.ones((len(labels), max_label_len), dtype=torch.long) * -100
        for i, l in enumerate(labels):
            padded_labels[i, :len(l)] = l

        # 리턴: Input, Dialect, Label (표준어 관련 리턴값 제거됨)
        return input_features, dialect_ids, padded_labels

import os
import re
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from transformers import WhisperProcessor, get_linear_schedule_with_warmup
from jiwer import cer
from tqdm import tqdm
from accelerate import Accelerator

from config.config import CFG
from dataset.dataset import KaldiMoEDataset, DataCollator
from model import DialectAwareWhisper


# ==========================================
# [Class] Distributed Smart Batch Sampler (수정됨)
# ==========================================
class DistributedSmartBatchSampler(Sampler):
    """
    Multi-GPU 환경에서 모든 Rank가 동일한 방식으로 Shuffle을 수행하도록 보장하는 Sampler
    """
    def __init__(self, data_source, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        if not hasattr(data_source, 'lengths'):
            raise ValueError("Dataset must have 'lengths' attribute for SmartBatching!")

        self.lengths = data_source.lengths
        self.total_size = len(self.data_source)

        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(range(len(self.data_source)))
        indices.sort(key=lambda i: self.lengths[i], reverse=True)

        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        my_indices = indices[self.rank : self.total_size : self.num_replicas]

        batches = [my_indices[i:i + self.batch_size] for i in range(0, len(my_indices), self.batch_size)]

        if self.shuffle:
            shuffle_idx = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in shuffle_idx]

        for batch in batches:
            yield batch

    def __len__(self):
        return len(range(0, self.num_samples, self.batch_size))

    def set_epoch(self, epoch):
        self.epoch = epoch


# ==========================================
# [Utils] Helper Functions
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def korean_normalizer(text):
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" ", "")
    return text.strip()

def evaluate_distributed(accelerator, model, dataloader, processor):
    """
    [수정] WER 대신 CER(음절 오류율)을 계산하여 정확한 성능 측정
    """
    model.eval()
    all_preds = []
    all_labels = []


    disable_tqdm = not accelerator.is_local_main_process
    iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=disable_tqdm)

    for batch in iterator:
        # [bf16 캐스팅] 검증 시 입력 타입 불일치 방지
        input_features = batch[0].to(device=accelerator.device, dtype=torch.bfloat16)
        dialect_ids = batch[1].to(accelerator.device)
        labels = batch[2].to(accelerator.device)

        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)

            # Generate 실행
            generated_ids = unwrapped_model.generate(
                input_features=input_features,
                dialect_ids=dialect_ids,
                max_new_tokens=128,
                num_beams=1,
                language="ko",
                task="transcribe",
                forced_decoder_ids=None # [중요] 설정 충돌 방지
            )

            # DDP Gather (Pad 채워서 길이 맞춤)
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=processor.tokenizer.pad_token_id)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            gathered_preds = accelerator.gather_for_metrics(generated_ids)
            gathered_labels = accelerator.gather_for_metrics(labels)

            if accelerator.is_main_process:
                # -100 라벨을 Pad 토큰으로 치환 (Decode 에러 방지)
                gathered_labels = torch.where(gathered_labels != -100, gathered_labels, processor.tokenizer.pad_token_id)

                # CPU로 이동 후 리스트에 저장
                all_preds.extend(gathered_preds.cpu().numpy())
                all_labels.extend(gathered_labels.cpu().numpy())

    cer_score = 0.0

    if accelerator.is_main_process and len(all_preds) > 0:
        # 1. 텍스트 디코딩 (특수 토큰 제거)
        decoded_preds = processor.batch_decode(all_preds, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(all_labels, skip_special_tokens=True)

        # 2. 정규화 (공백 제거 포함)
        decoded_preds = [korean_normalizer(text) for text in decoded_preds]
        decoded_labels = [korean_normalizer(text) for text in decoded_labels]

        cer_score = cer(decoded_labels, decoded_preds)

    return cer_score

def train():
    accelerator = Accelerator(mixed_precision="bf16")
    save_dir = "exp"
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")

    set_seed(CFG['seed'])

    # 1. Data & Model Prep
    processor = WhisperProcessor.from_pretrained(CFG['whisper_model'])

    # Train Dataset
    train_dataset = KaldiMoEDataset(
        wav_scp=CFG['wav_scp'],
        text_file=CFG['text_file'],
        processor=processor
    )

    # [수정됨] 위에서 정의한 DistributedSmartBatchSampler 사용
    smart_sampler = DistributedSmartBatchSampler(
        train_dataset,
        batch_size=CFG['batch_size'],
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=CFG['seed'],
    )

    # batch_sampler 옵션을 쓰므로 batch_size, shuffle 인자 제거
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=smart_sampler,
        collate_fn=DataCollator(),
        num_workers=CFG.get('num_workers', 4),
        pin_memory=True
    )

    # Validation Dataset
    val_dataset = KaldiMoEDataset(
        wav_scp=CFG['val_wav_scp'],
        text_file=CFG['val_text_file'],
        processor=processor
    )
    if not train_dataset or not val_dataset:
        raise ValueError("Training and validation lists must contain matching utterance IDs.")

    val_dataloader = DataLoader(
        val_dataset, batch_size=CFG['val_batch_size'], shuffle=False,
        collate_fn=DataCollator(), num_workers=CFG.get('num_workers', 4), pin_memory=True
    )

    model = DialectAwareWhisper(CFG)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.get('lr_experts', 1e-3),
        weight_decay=CFG.get('weight_decay', 0.01)
    )

    # 전체 스텝 수
    total_steps = len(train_dataloader) * CFG['epochs'] * 3
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    # 2. Accelerate Prepare
    # [중요] train_dataloader는 넣지 않음 (이중 분할 방지)
    model, optimizer, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, val_dataloader, lr_scheduler
    )

    best_cer = float('inf')
    early_stopping_counter = 0
    patience_limit = CFG.get('patience', 5)
    stop_training = False # [수정] 학습 중단 플래그 추가

    for epoch in range(CFG['epochs']):
        # [핵심] Epoch마다 Sampler의 Seed를 변경해줌 (모든 GPU 동기화)
        smart_sampler.set_epoch(epoch)
        model.train()

        # [수정] Epoch별 누적 Loss 초기화
        epoch_total_loss = 0.0
        valid_step_count = 0

        disable_tqdm = not accelerator.is_local_main_process
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", disable=disable_tqdm)

        for step, batch in enumerate(pbar):
            # [중요] Prepare를 안 했으므로 수동으로 GPU 할당
            input_features = batch[0].to(accelerator.device)
            dialect_ids = batch[1].to(accelerator.device)
            labels = batch[2].to(accelerator.device)

            with accelerator.accumulate(model):
                outputs = model(
                    input_features=input_features,
                    dialect_ids=dialect_ids,
                    labels=labels
                )
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        print(f"\n⚠️ [Warning] NaN/Inf Loss at step {step} (Epoch {epoch+1})")
                        print(f"   Main Loss: {loss.item():.4f}")
                        print("   -> Skipping Optimizer Step (Weights not updated)")
                    optimizer.zero_grad()
                    continue

                # item()을 호출하여 스칼라 값으로 변환 후 누적
                current_loss_val = loss.item()
                epoch_total_loss += current_loss_val
                valid_step_count += 1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 10 == 0:
                average_loss = epoch_total_loss / valid_step_count
                pbar.set_postfix(avg_loss=average_loss, loss=current_loss_val)

        # =======================================================
        # Validation Loop
        # =======================================================
        if accelerator.is_main_process:
            print(f"\n>>> Validation Epoch {epoch+1}...")

        accelerator.wait_for_everyone()

        current_cer = evaluate_distributed(accelerator, model, val_dataloader, processor)

        if accelerator.is_main_process:
            print(f"   [Epoch {epoch+1}] Val CER: {current_cer*100:.2f}%")

            if current_cer < best_cer:
                best_cer = current_cer
                early_stopping_counter = 0
                unwrapped_model = accelerator.unwrap_model(model)
                best_save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(unwrapped_model.state_dict(), best_save_path)
                print(f"   🌟 New Best Model Saved: {best_save_path} ({best_cer*100:.2f}%)")
            else:
                early_stopping_counter += 1
                print(f"   🛑 No improvement ({early_stopping_counter}/{patience_limit})")

                # [수정] Patience 도달 시 중단 플래그 설정
                if early_stopping_counter >= patience_limit:
                    print(f"\n✋ Early Stopping Triggered! (No improvement for {patience_limit} epochs)")
                    stop_training = True

            # Regular Checkpoint Save
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(save_dir, f"moe_whisper_ep{epoch+1}.pth")
            torch.save(unwrapped_model.state_dict(), save_path)

        # [수정] 모든 GPU에 중단 신호 동기화 (Broadcast)
        # 메인 프로세스의 stop_training 값을 텐서로 변환하여 전파
        stop_signal = torch.tensor(1 if stop_training else 0, device=accelerator.device)
        if accelerator.num_processes > 1:
            dist.broadcast(stop_signal, src=0)

        # 모든 GPU가 동기화된 후 루프 탈출
        if stop_signal.item() == 1:
            if accelerator.is_main_process:
                print("🚀 Training stopped by Early Stopping.")
            break

        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    train()

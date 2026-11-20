import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

class DynamicLoRALinear(nn.Module):
    """
    MoE-LoRA의 안정성을 채용한 Dynamic LoRA Layer
    """
    def __init__(self, original_layer, noise_dim, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer # Frozen Layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.rank = rank
        self.scaling = alpha / rank

        # [안전장치 1] Input Normalization
        # 잡음 벡터의 스케일을 일정하게 유지
        self.noise_norm = nn.LayerNorm(noise_dim)

        # -----------------------------------------------------------
        # Hypernetwork (Split Structure for Stability)
        # -----------------------------------------------------------
        # 하나의 큰 네트워크 대신 A와 B를 만드는 네트워크를 분리하여
        # B를 확실하게 0으로 제어합니다.

        # Generator A: (Noise) -> (In * Rank)
        # Kaiming Init과 유사하게 랜덤하게 초기화 (표현력 확보)
    # Generator A: (Noise) -> (In * Rank)
        self.gen_A = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # [추가] 과적합 방지 핵심
            nn.Linear(128, self.in_features * rank)
        )

        # Generator B: (Noise) -> (Rank * Out)
        self.gen_B = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # [추가] 과적합 방지 핵심
            nn.Linear(128, self.rank * self.out_features)
        )

        # Init Weights
        # A는 랜덤 (표현력)
        nn.init.normal_(self.gen_A[-1].weight, std=0.02)

        # B는 제로 (안전성 - MoE의 lora_B=0 로직과 동일)
        nn.init.zeros_(self.gen_B[-1].weight)
        nn.init.zeros_(self.gen_B[-1].bias)

        # 상태 저장용
        self.current_noise_emb = None

    def forward(self, x):
        # 1. Original Frozen Output
        # Base Model은 건드리지 않음
        out_orig = self.original_layer(x)

        # 잡음 정보가 없으면 원본만 반환
        if self.current_noise_emb is None:
            return out_orig

        # 2. Hypernetwork Forward
        # 정규화된 잡음 입력
        norm_noise = self.noise_norm(self.current_noise_emb)

        # Generate Weights
        bsz = x.size(0)

        params_A = self.gen_A(norm_noise) # (Batch, In*Rank)
        params_B = self.gen_B(norm_noise) # (Batch, Rank*Out) -> 초기엔 0

        # 3. Reshape
        matrix_A = params_A.view(bsz, self.in_features, self.rank)
        matrix_B = params_B.view(bsz, self.rank, self.out_features)

        # 4. Compute Delta (x @ A @ B)
        # x: (Batch, Seq, In)
        hidden = torch.bmm(x, matrix_A.to(x.dtype))
        delta = torch.bmm(hidden, matrix_B.to(x.dtype))


        # 5. Add to Original
        return out_orig + (delta * self.scaling)

class DialectAwareWhisper(nn.Module):
    def __init__(self, cfg, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg

        # 1. Whisper 로드
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            cfg['whisper_model'],
            attn_implementation="sdpa", # <--- Flash Attention 켜기
            dtype=dtype,
        )
        self.whisper.config.use_cache = False

        # 2. [핵심] 방언 임베딩 레이어 (6개 방언)
        # noise_dim: Dynamic LoRA에 주입될 벡터 크기 (예: 64)
        self.noise_dim = cfg.get('noise_dim', 64)
        self.num_dialects = 6

        self.dialect_emb = nn.Embedding(self.num_dialects, self.noise_dim)

        # 임베딩 초기화 (안정적인 학습을 위해)
        nn.init.normal_(self.dialect_emb.weight, std=0.02)

        # 3. Dynamic LoRA 주입
        print(f">>> Injecting Dynamic LoRA (Rank={cfg.get('lora_rank', 8)})...")
        self.inject_dynamic_lora()

        # 4. Freeze 설정
        self.setup_freeze()

    def inject_dynamic_lora(self):
        for layer in self.whisper.model.encoder.layers:
            attn = layer.self_attn
            # q_proj, v_proj 교체
            attn.q_proj = DynamicLoRALinear(
                attn.q_proj, self.noise_dim, rank=self.cfg.get('lora_rank', 8)
            )
            attn.v_proj = DynamicLoRALinear(
                attn.v_proj, self.noise_dim, rank=self.cfg.get('lora_rank', 8)
            )

    def setup_freeze(self):
        # 1. 전체 Freeze
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 2. Embedding Layer 학습
        for param in self.dialect_emb.parameters():
            param.requires_grad = True

        # 3. Dynamic LoRA Generator 학습
        trainable_cnt = 0
        for module in self.whisper.modules():
            if isinstance(module, DynamicLoRALinear):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_cnt += param.numel()

        print(f"🔥 Model Ready. Trainable Params: {trainable_cnt:,}")

    def forward(self, input_features, dialect_ids, labels=None):
        # 1. ID -> Vector 변환
        # dialect_ids: (Batch,)
        dialect_vector = self.dialect_emb(dialect_ids) # (Batch, noise_dim)

        # 2. LoRA 모듈에 벡터 주입 (Broadcasting)
        for module in self.whisper.modules():
            if isinstance(module, DynamicLoRALinear):
                module.current_noise_emb = dialect_vector

                # [Tip] 학습 시 Overfitting 방지를 위한 Jittering
                # 지울지 말지 고민 필요
                if self.training:
                    noise = torch.randn_like(dialect_vector) * 0.1
                    module.current_noise_emb += noise

        # 3. Whisper Forward
        return self.whisper(input_features=input_features, labels=labels)

    def generate(self, input_features, dialect_ids, **kwargs):
        # Inference 시 사용
        dialect_vector = self.dialect_emb(dialect_ids)

        for module in self.whisper.modules():
            if isinstance(module, DynamicLoRALinear):
                module.current_noise_emb = dialect_vector

        return self.whisper.generate(input_features=input_features, **kwargs)

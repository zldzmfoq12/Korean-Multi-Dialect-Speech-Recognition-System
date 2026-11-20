import argparse
import re
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperProcessor

from config.config import CFG
from model import DialectAwareWhisper


def transcribe(audio_path, dialect_id, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter_paths = [
        key.removeprefix("whisper.").removesuffix(".original_layer.weight")
        for key in state
        if re.fullmatch(
            r"whisper.model.(encoder.layers.d+.self_attn|"
            r"decoder.layers.d+.(self_attn|encoder_attn)).[qv]_proj.original_layer.weight",
            key,
        )
    ]
    if not adapter_paths:
        raise ValueError("The checkpoint does not contain Whisper Dynamic LoRA weights.")

    # The LRL head is used only for the auxiliary training loss.
    inference_state = {key: weight for key, weight in state.items() if not key.startswith("lrl_head.")}
    processor = WhisperProcessor.from_pretrained(CFG["whisper_model"])
    model = DialectAwareWhisper(CFG, dtype=dtype, adapter_paths=adapter_paths)
    model.load_state_dict(inference_state)
    model.to(device).eval()

    wave, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        wave = torchaudio.functional.resample(wave, sample_rate, 16000)
    wave = wave.mean(dim=0)
    input_features = processor(
        wave.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=dtype)
    dialect_ids = torch.tensor([dialect_id], device=device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=input_features,
            dialect_ids=dialect_ids,
            language="ko",
            task="transcribe",
            num_beams=1,
            max_new_tokens=128,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe Korean speech using a dialect ID.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--dialect", type=int, choices=range(6), default=0)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.audio, args.ckpt):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    print(transcribe(args.audio, args.dialect, args.ckpt))

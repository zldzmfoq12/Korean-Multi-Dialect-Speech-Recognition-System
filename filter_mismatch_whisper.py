"""Filter prepared utterance lists using Faster-Whisper ASR confidence."""

import argparse
import math
from pathlib import Path

from tqdm import tqdm

from prepare_data import read_manifest


def filter_manifests(args):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    audio_entries = read_manifest(args.wav_scp)
    transcripts = read_manifest(args.text)
    model = WhisperModel(
        args.model, device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
    )
    pipeline = BatchedInferencePipeline(model)
    output = args.output_dir.resolve()
    destinations = (output / "wav.scp", output / "text")
    inputs = {args.wav_scp.resolve(), args.text.resolve()}
    if any(path in inputs for path in destinations):
        raise ValueError("Use a separate output directory for the filtered manifests.")
    output.mkdir(parents=True, exist_ok=True)

    kept = failed = 0
    with (
        destinations[0].open("w", encoding="utf-8") as wav_list,
        destinations[1].open("w", encoding="utf-8") as text_list,
    ):
        for key, filename in tqdm(audio_entries.items(), desc="Filtering audio", unit="file"):
            if key not in transcripts:
                continue
            try:
                segments, _ = pipeline.transcribe(filename, language="ko", batch_size=args.batch_size)
                log_probs = [segment.avg_logprob for segment in segments]
            except (OSError, RuntimeError, ValueError):
                failed += 1
                continue
            if not log_probs or math.exp(sum(log_probs) / len(log_probs)) < args.threshold:
                continue
            wav_list.write(f"{key} {filename}\n")
            text_list.write(f"{key} {transcripts[key]}\n")
            kept += 1
    print(f"Kept {kept} of {len(audio_entries)} clips. Transcription failures: {failed}")
    if not kept:
        raise SystemExit("No clips passed confidence filtering.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1:
        parser.error("threshold must be in [0, 1] and batch size must be positive")
    filter_manifests(args)

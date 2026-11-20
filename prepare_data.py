"""Prepare local WAV segments and Kaldi-style manifests for dialect ASR."""

import argparse
from collections import Counter
import json
from math import gcd
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from tqdm import tqdm


SAMPLE_RATE = 16_000
DIALECT_REGIONS = {
    "se": ("표준", "수도권", "서울", "경기", "인천"),
    "jl": ("전라", "전남", "전북", "광주"),
    "gs": ("경상", "경남", "경북", "부산", "대구", "울산"),
    "cc": ("충청", "충남", "충북", "대전", "세종"),
    "gw": ("강원",),
    "jj": ("제주",),
}


def read_manifest(path):
    entries = {}
    if path is None:
        return entries
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split(maxsplit=1)
            if not fields:
                continue
            if len(fields) != 2 or not fields[1].strip() or fields[0] in entries:
                raise ValueError(f"Invalid or duplicate entry at {path}:{number}")
            entries[fields[0]] = fields[1].strip()
    return entries


def dialect_tag(region):
    region = str(region or "").strip().lower()
    for tag, names in DIALECT_REGIONS.items():
        if region == tag or any(name in region for name in names):
            return tag
    return None


def metadata_region(metadata):
    speakers = metadata.get("speaker")
    speaker = speakers[0] if isinstance(speakers, list) and speakers else {}
    return (
        metadata.get("화자정보", {}).get("Region")
        or metadata.get("대화정보", {}).get("cityCode")
        or speaker.get("residenceProvince")
        or metadata.get("collector", {}).get("residenceProvince")
    )


def seconds(timestamp):
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    hours, minutes, seconds_part = timestamp.split(":")
    return 3600 * int(hours) + 60 * int(minutes) + float(seconds_part)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def segments(key, transcript, metadata, length, sample_rate, transcript_field, default_dialect):
    tag = dialect_tag(metadata_region(metadata)) or default_dialect
    sentences = metadata.get("transcription", {}).get("sentences") or []
    if not sentences:
        fields = transcript.split(maxsplit=1)
        if len(fields) == 2 and fields[0] in DIALECT_REGIONS:
            tag, transcript = fields
        yield key, tag, normalize(transcript), 0, length
        return

    for index, sentence in enumerate(sentences, 1):
        start = seconds(sentence["startTime"])
        end = seconds(sentence["endTime"])
        if end <= start:
            continue
        # Match the segmentation margin used by the original preparation code.
        left = max(0, int((start - 0.5) * sample_rate))
        right = min(length, int((end + 0.5) * sample_rate))
        yield f"{key}_{index:03d}", tag, normalize(sentence.get(transcript_field)), left, right


def prepare(args):
    wavs = read_manifest(args.wav_scp)
    transcripts = read_manifest(args.text)
    labels = read_manifest(args.label_scp)
    output = args.output_dir.resolve()
    if any(character.isspace() for character in str(output)):
        raise ValueError("Use an output directory with no whitespace in its path.")
    destinations = (output / "wav.scp", output / "text")
    inputs = {path.resolve() for path in (args.wav_scp, args.text, args.label_scp) if path is not None}
    if any(path in inputs for path in destinations):
        raise ValueError("Use a separate output directory for the prepared manifests.")
    clips = output / "wavs"
    clips.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    written = set()
    with (
        destinations[0].open("w", encoding="utf-8") as wav_list,
        destinations[1].open("w", encoding="utf-8") as text_list,
    ):
        for key, filename in tqdm(wavs.items(), desc="Preparing audio", unit="file"):
            if not re.fullmatch(r"[\w.-]+", key) or key in {".", ".."}:
                raise ValueError(f"Invalid utterance ID: {key!r}")
            path = Path(filename)
            label = Path(labels[key]) if key in labels else path.with_suffix(".json")
            try:
                metadata = json.loads(label.read_text(encoding="utf-8-sig")) if label.is_file() else {}
                if not isinstance(metadata, dict):
                    raise ValueError("The label JSON must be an object.")
                audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
                intervals = list(segments(
                    key, transcripts.get(key, ""), metadata, len(audio), sample_rate,
                    args.transcript_field, args.default_dialect,
                ))
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                stats["invalid_source"] += 1
                continue

            for segment_id, tag, transcript, start, end in intervals:
                if not tag or not transcript:
                    stats["missing_label"] += 1
                    continue
                if not 0.1 * sample_rate <= end - start <= 30 * sample_rate:
                    stats["invalid_duration"] += 1
                    continue
                clip = audio[start:end].mean(axis=1)
                if not np.isfinite(clip).all() or np.mean(clip ** 2) < 10 ** (-50 / 10):
                    stats["silent_or_invalid"] += 1
                    continue
                if segment_id in written:
                    raise ValueError(f"Duplicate output utterance ID: {segment_id}")
                if sample_rate != SAMPLE_RATE:
                    divisor = gcd(sample_rate, SAMPLE_RATE)
                    clip = resample_poly(clip, SAMPLE_RATE // divisor, sample_rate // divisor)
                destination = clips / f"{segment_id}.wav"
                if destination.resolve() == path.resolve():
                    raise ValueError(f"Refusing to overwrite source audio for {key}")
                sf.write(destination, clip, SAMPLE_RATE, subtype="PCM_16")
                wav_list.write(f"{segment_id} {destination.as_posix()}\n")
                text_list.write(f"{segment_id} {tag} {transcript}\n")
                written.add(segment_id)
    print(f"Kept {len(written)} clips. Rejected: {dict(stats)}")
    if not written:
        raise SystemExit("No clips passed preparation. Check the input manifests and dialect labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", type=Path, required=True)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--label-scp", type=Path, help="Optional utterance ID to JSON path mapping")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transcript-field", default="dialect")
    parser.add_argument("--default-dialect", choices=DIALECT_REGIONS)
    prepare(parser.parse_args())

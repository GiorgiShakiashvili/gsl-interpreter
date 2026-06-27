from __future__ import annotations

import argparse
import os
import sys

from gsl_interpreter.tts import DEFAULT_GEORGIAN_VOICE, DEFAULT_TTS_MODE, TTS_MODE_CHOICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gsl_interpreter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record webcam samples for a label")
    record.add_argument("--label", required=True, help="UTF-8 label text")
    record.add_argument("--samples", required=True, type=int, help="Number of samples to save")
    record.add_argument("--signer", required=True, help="Signer identifier used in filenames")
    record.add_argument("--camera", default=0, type=int, help="OpenCV camera index")

    train = subparsers.add_parser("train", help="Train a classifier from recorded samples")
    train.add_argument("--model-out", default="models/classifier.pkl", help="Output model path")

    infer = subparsers.add_parser("infer", help="Run live webcam inference")
    infer.add_argument("--model", default="models/classifier.pkl", help="Model artifact path")
    infer.add_argument("--camera", default=0, type=int, help="OpenCV camera index")
    infer.add_argument("--width", default=960, type=int, help="Requested camera width")
    infer.add_argument("--height", default=540, type=int, help="Requested camera height")
    infer.add_argument("--camera-buffer", default=1, type=int, help="Requested camera buffer size")
    infer.add_argument(
        "--sentence-log",
        default="data/sentences.jsonl",
        help="JSONL file where recognized sentence updates are saved",
    )
    infer.add_argument(
        "--no-autosave-sentences",
        dest="autosave_sentences",
        action="store_false",
        help="Disable backup sentence logging after each accepted sign",
    )
    infer.add_argument(
        "--autosave-sentences",
        dest="autosave_sentences",
        action="store_true",
        help="Enable backup sentence logging after each accepted sign",
    )
    infer.set_defaults(autosave_sentences=False)
    infer.add_argument(
        "--start-threshold",
        default=4.5,
        type=float,
        help="Maximum start-pose distance accepted before motion capture begins",
    )
    infer.add_argument(
        "--tracking-complexity",
        default=1,
        choices=(0, 1, 2),
        type=int,
        help="MediaPipe pose model complexity; 1 matches recording/training, 0 is fastest",
    )
    infer.add_argument(
        "--tts-mode",
        default=DEFAULT_TTS_MODE,
        choices=TTS_MODE_CHOICES,
        help="Georgian speech mode: recognized words, saved sentences, both, or off",
    )
    infer.add_argument(
        "--tts-voice",
        default=DEFAULT_GEORGIAN_VOICE,
        help="Microsoft Georgian neural voice used by edge-tts",
    )
    infer.add_argument(
        "--tts-rate",
        default="+0%",
        help='Speech rate passed to edge-tts, for example "+10%%" or "-10%%"',
    )
    infer.add_argument(
        "--tts-volume",
        default="+0%",
        help='Speech volume passed to edge-tts, for example "+0%%"',
    )
    infer.add_argument(
        "--tts-cache-dir",
        default="data/tts",
        help="Directory for cached Georgian TTS MP3 files",
    )

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()

    if args.command == "record":
        from gsl_interpreter.dataset import record_samples

        record_samples(args.label, args.samples, args.signer, camera_index=args.camera)
    elif args.command == "train":
        from gsl_interpreter.train import train_model

        train_model(args.model_out)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif args.command == "infer":
        from gsl_interpreter.infer import run_inference

        run_inference(
            args.model,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            camera_buffer=args.camera_buffer,
            start_threshold=args.start_threshold,
            tracking_complexity=args.tracking_complexity,
            sentence_log=args.sentence_log,
            autosave_sentences=args.autosave_sentences,
            tts_mode=args.tts_mode,
            tts_voice=args.tts_voice,
            tts_rate=args.tts_rate,
            tts_volume=args.tts_volume,
            tts_cache_dir=args.tts_cache_dir,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

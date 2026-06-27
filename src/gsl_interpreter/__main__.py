from __future__ import annotations

import argparse
import os
import sys


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
        "--tracking-complexity",
        default=0,
        choices=(0, 1, 2),
        type=int,
        help="MediaPipe pose model complexity; 0 is fastest, 1/2 can improve tracking",
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
            tracking_complexity=args.tracking_complexity,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""카메라 자세 인식 결과를 외부 디스플레이 표정으로 보여준다.

카메라와 디스플레이만 필요하다.

예시:
    python scripts/run_display.py --camera 0
    python scripts/run_display.py --camera /dev/video2 --fullscreen
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_stream import CameraStream
from src.display.expression_display import (
    DisplayPolicy,
    ExpressionController,
    ExpressionDisplay,
)
from src.perception.calibration import load_reference
from src.perception.pose_estimator import PoseEstimator
from src.perception.posture_features import (
    MedianFilter,
    PostureAngles,
    apply_reference,
    clamp_angles,
    extract_angles,
    smooth,
)
from src.posture.classifier import classify


CONFIG_PATH = PROJECT_ROOT / "config" / "posture.yaml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def run(args):
    config = load_config()
    perception = config["perception"]
    angle_config = config["angles"]
    display_config = config.get("display") or {}

    source = args.camera if args.camera is not None else perception.get("camera", 0)
    width = int(args.width or perception.get("width", 640))
    height = int(args.height or perception.get("height", 480))
    reference = load_reference() if perception.get("use_saved_reference", False) else None

    policy = DisplayPolicy(
        bad_sustain_seconds=float(
            args.bad_seconds if args.bad_seconds is not None else
            display_config.get("bad_sustain_seconds", 3.0)),
        recovery_sustain_seconds=float(
            display_config.get("recovery_sustain_seconds", 1.0)),
    )
    controller = ExpressionController(policy)
    display = ExpressionDisplay(
        width=int(args.display_width or display_config.get("width", 1024)),
        height=int(args.display_height or display_config.get("height", 600)),
        fullscreen=args.fullscreen,
    )
    median = MedianFilter(angle_config.get("median_window", 7))
    previous = None
    last_expression = controller.expression
    last_label = "unknown"
    started = time.monotonic()

    try:
        with CameraStream(source, width, height) as camera, \
                PoseEstimator(args.model or perception.get("model")) as estimator:
            print("NOTIFYI 디스플레이 모드 시작")
            print("카메라 자세 인식 → 표정 표시. q 또는 Esc로 종료합니다.")
            while True:
                frame = camera.read()
                now = time.monotonic()
                if frame is None:
                    posture = classify(PostureAngles(now, 0.0, 0.0, 0.0))
                    expression = controller.update(posture, now)
                    key = display.show(expression, posture.label)
                else:
                    landmarks, world = estimator.detect(
                        frame, (now - started) * 1000.0)
                    measured = None
                    if landmarks and world:
                        measured = extract_angles(
                            world, landmarks, now,
                            perception["min_visibility"],
                            perception["invert_torso"],
                            perception["invert_neck"],
                        )

                    if measured is not None:
                        measured = median.apply(measured)
                        measured = apply_reference(measured, reference)
                        measured = clamp_angles(
                            measured,
                            angle_config["max_torso_pitch_deg"],
                            angle_config["max_neck_pitch_deg"],
                        )
                        measured = smooth(
                            previous,
                            measured,
                            angle_config["smoothing_alpha"],
                            angle_config.get("neck_smoothing_alpha"),
                        )
                        previous = measured
                        posture = classify(measured)
                    else:
                        previous = None
                        median.reset()
                        posture = classify(PostureAngles(now, 0.0, 0.0, 0.0))

                    expression = controller.update(posture, now)
                    if expression != last_expression or posture.label != last_label:
                        print(f"[display] {posture.label} -> {expression.value}")
                        last_expression = expression
                        last_label = posture.label
                    key = display.show(
                        expression,
                        posture.label,
                        measured.torso_pitch_deg if measured else None,
                        measured.neck_pitch_deg if measured else None,
                    )

                if key in (ord("q"), 27):
                    break
    finally:
        display.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NOTIFYI display mode")
    parser.add_argument("--camera", help="카메라 인덱스 또는 /dev/video 경로")
    parser.add_argument("--model", help="lite / full / heavy 또는 .task 경로")
    parser.add_argument("--width", type=int, help="카메라 입력 너비")
    parser.add_argument("--height", type=int, help="카메라 입력 높이")
    parser.add_argument("--display-width", type=int, help="표정 창 너비")
    parser.add_argument("--display-height", type=int, help="표정 창 높이")
    parser.add_argument("--bad-seconds", type=float,
                        help="나쁜 자세 지속 시간(초)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="표정 창을 전체 화면으로 표시")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

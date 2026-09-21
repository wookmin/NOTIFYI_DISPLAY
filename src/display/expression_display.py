"""자세 상태를 단순한 로봇 표정으로 보여주는 디스플레이 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from src.posture.classifier import PostureState


class Expression(str, Enum):
    """디스플레이에 표시할 표정."""

    SMILE = "smile"
    FROWN = "frown"


@dataclass(frozen=True)
class DisplayPolicy:
    """표정 전환에 필요한 안정화 시간."""

    bad_sustain_seconds: float = 3.0
    recovery_sustain_seconds: float = 1.0


class ExpressionController:
    """자세 분류를 표정 상태로 바꾼다.

    잠깐 흔들린 프레임 때문에 표정이 바로 바뀌지 않도록 나쁜 자세와
    회복 모두 지속시간을 요구한다. 사람을 놓친 프레임은 나쁜 자세로
    간주하지 않고 현재 표정을 유지한다.
    """

    def __init__(self, policy: DisplayPolicy | None = None):
        self.policy = policy or DisplayPolicy()
        self.expression = Expression.SMILE
        self._bad_since: float | None = None
        self._good_since: float | None = None

    def update(self, posture: PostureState, now: float) -> Expression:
        """최신 자세를 반영하고 현재 표정을 반환한다."""
        if posture.label == "unknown":
            self._bad_since = None
            self._good_since = None
            return self.expression

        if posture.is_bad:
            self._good_since = None
            if self.expression is Expression.FROWN:
                return self.expression
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= self.policy.bad_sustain_seconds:
                self.expression = Expression.FROWN
                self._bad_since = None
            return self.expression

        # 좋은 자세가 회복된 경우
        self._bad_since = None
        if self.expression is Expression.SMILE:
            self._good_since = None
            return self.expression
        if self._good_since is None:
            self._good_since = now
        if now - self._good_since >= self.policy.recovery_sustain_seconds:
            self.expression = Expression.SMILE
            self._good_since = None
        return self.expression


class ExpressionDisplay:
    """OpenCV 창에 표정만 그린다. 카메라 영상은 표시하지 않는다."""

    def __init__(self, width: int = 1024, height: int = 600,
                 fullscreen: bool = False,
                 window_name: str = "NOTIFYI Display"):
        self.width = int(width)
        self.height = int(height)
        self.fullscreen = fullscreen
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        if fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, expression: Expression, posture_label: str,
             torso_pitch: float | None = None,
             neck_pitch: float | None = None) -> int:
        """표정을 그리고 키 입력을 반환한다. q/Esc로 종료한다."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background = (35, 55, 75) if expression is Expression.SMILE else (55, 35, 45)
        canvas[:, :] = background

        center = (self.width // 2, self.height // 2 - 15)
        face_radius = max(120, min(self.width, self.height) // 3)
        cv2.circle(canvas, center, face_radius, (70, 215, 245), -1)
        cv2.circle(canvas, center, face_radius, (20, 35, 45), 5)

        eye_y = center[1] - face_radius // 3
        eye_dx = face_radius // 2
        for eye_x in (center[0] - eye_dx, center[0] + eye_dx):
            cv2.circle(canvas, (eye_x, eye_y), max(10, face_radius // 12),
                       (20, 35, 45), -1)

        brow_y = eye_y - face_radius // 4
        if expression is Expression.SMILE:
            cv2.line(canvas, (center[0] - eye_dx - 25, brow_y),
                     (center[0] - eye_dx + 25, brow_y), (20, 35, 45), 8)
            cv2.line(canvas, (center[0] + eye_dx - 25, brow_y),
                     (center[0] + eye_dx + 25, brow_y), (20, 35, 45), 8)
            cv2.ellipse(canvas, (center[0], center[1] + face_radius // 5),
                        (face_radius // 2, face_radius // 3), 0, 15, 165,
                        (20, 35, 45), 10)
            title = "GOOD POSTURE"
        else:
            # 안쪽이 낮아지는 눈썹으로 걱정스러운 표정을 만든다.
            cv2.line(canvas, (center[0] - eye_dx - 25, brow_y + 22),
                     (center[0] - eye_dx + 25, brow_y - 12), (20, 35, 45), 8)
            cv2.line(canvas, (center[0] + eye_dx - 25, brow_y - 12),
                     (center[0] + eye_dx + 25, brow_y + 22), (20, 35, 45), 8)
            cv2.ellipse(canvas, (center[0], center[1] + face_radius // 2),
                        (face_radius // 2, face_radius // 3), 0, 195, 345,
                        (20, 35, 45), 10)
            title = "PLEASE SIT STRAIGHT"

        cv2.putText(canvas, title, (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (245, 245, 245), 2, cv2.LINE_AA)
        detail = posture_label.upper()
        if torso_pitch is not None and neck_pitch is not None:
            detail += f"  torso {torso_pitch:+.1f}  neck {neck_pitch:+.1f}"
        cv2.putText(canvas, detail, (40, self.height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (215, 225, 230), 1,
                    cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)

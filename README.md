# NOTIFYI

Logitech 카메라로 사용자의 자세를 판독하고, 외부 디스플레이에 표정을 보여주는
프로그램입니다.

```text
카메라 → MediaPipe 자세 인식 → 자세 분류 → 표정 제어 → 디스플레이
```

이 버전은 카메라와 디스플레이만 사용합니다. OpenCR, 배터리, Dynamixel 모터,
음성 기능, 카메라 화면 미러링은 사용하지 않습니다.

## 실행

프로젝트 폴더에서 필요한 패키지를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

자세 인식 모델은 다음 위치에 있어야 합니다.

```text
models/pose/pose_landmarker_lite.task
```

Logitech 카메라를 기본 장치로 실행합니다.

```bash
python scripts/run_display.py --camera 0
```

Ubuntu에서 카메라 장치가 `/dev/video2`라면 다음처럼 실행합니다.

```bash
python scripts/run_display.py --camera /dev/video2 --fullscreen
```

표정 창을 디스플레이로 옮긴 다음 전체 화면으로 사용하면 됩니다. `q` 또는
`Esc`를 누르면 종료합니다.

## 표정 규칙

- 정상 자세: 웃는 표정
- 나쁜 자세가 기본 3초 이상 지속: 찡그린 표정
- 다시 정상 자세가 1초 이상 지속: 웃는 표정
- 사람이 잠시 감지되지 않으면 기존 표정을 유지하고, 다시 감지되면 판독을 계속함

지속시간은 `config/posture.yaml`의 `display` 항목에서 조정할 수 있습니다.

## 테스트

```bash
pytest -q
```

원본 POSTIC 프로젝트는 `/Users/im_inwook/dev/POSTIC`에 그대로 보존되어 있고,
이 폴더는 디스플레이 방식에 필요한 코드만 담고 있습니다.

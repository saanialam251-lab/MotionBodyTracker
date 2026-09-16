# MotionBodyTracker
# Motion + Body Tracker

Native **Kotlin / Jetpack Compose** Android app plus a **Python webcam** version.

The app watches the camera, draws a green skeleton with red joints, finds moving areas, and reacts to simple hand gestures.

## Features

- Front + back camera (Flip)
- Hand tracking — up to 2 hands, 21 landmarks each (MediaPipe)
- Full-body pose tracking — 33 landmarks (MediaPipe)
- Motion detection — luma frame differencing, dashed box, glow, live sensitivity slider
- Gestures
  - **Open palm** → start tracking
  - **Peace sign** → snapshot
  - **Fist** → stop tracking
- Snapshot — saves camera frame + overlay to `Pictures/MotionTracker`
- Animated UI — breathing background, motion bar, pulsing alert, sliding log
- Laptop version in `python/tracker.py`

## Requirements

- Android Studio Hedgehog or newer
- Phone or emulator with a camera (real device recommended)
- Min SDK 24, Target SDK 34
- For the Python app: Python 3.9+, a webcam

## Android setup

1. Open this folder in Android Studio and let Gradle sync.
2. Confirm both model files are in `app/src/main/assets/`:
   - `hand_landmarker.task`
   - `pose_landmarker.task`
3. Run on a device. Grant camera permission. Press **Start**.

## Python setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tracker.py

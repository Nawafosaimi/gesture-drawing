# Gesture-Based Smart Drawing System

A **product-grade**, real-time gesture-based drawing system that lets you draw in the air using hand gestures captured by a webcam.

Built with Python, OpenCV, MediaPipe, and NumPy — designed as a professional AI/Computer Vision portfolio project.

---

## ✨ Features

| Feature | Description |
|---|---|
| ✋ Real-time hand tracking | 21-landmark hand detection via MediaPipe |
| ✏️ Air drawing | Draw with your index finger in the air |
| 🎨 Color palette | Cycle through 8 colors with a gesture |
| 📏 Brush size control | Thumbs up/down to adjust thickness |
| 🧹 Eraser mode | V-sign gesture toggles eraser |
| ↩️ Undo/Redo | Undo last stroke with a gesture or keyboard |
| 🗑️ Clear canvas | Hold fist for 1.5s to clear (prevents accidental clear) |
| 💾 Save drawing | OK-sign gesture or press S to save as PNG |
| 🎯 Kalman-filtered tracking | Smooth, jitter-free drawing |
| 🛡️ Debounced gesture FSM | No accidental mode switches |

---

## 🏗️ Architecture

```
Camera → HandTracker → GestureEngine → StateManager → DrawingEngine → UIRenderer → Display
                                            ↑                              ↑
                                      KalmanFilter                   ExportManager
```

| Module | File | Role |
|---|---|---|
| Config | `config/settings.py` | All tunable constants |
| Hand Tracker | `core/hand_tracker.py` | MediaPipe Hands wrapper |
| Gesture Engine | `core/gesture_engine.py` | Hand pose → gesture classification |
| State Manager | `core/state_manager.py` | FSM with debounce & confidence gating |
| Smoothing | `core/smoothing.py` | Kalman filter + EMA for coordinates |
| Drawing Engine | `core/drawing_engine.py` | Stroke list, canvas, undo/redo |
| UI Renderer | `core/ui_renderer.py` | HUD, toolbar, landmarks, instructions |
| Export Manager | `core/export_manager.py` | PNG save, video recording |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aircavas.git
cd app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

### CLI Options

```
python main.py --source video.mp4     # Use a video file instead of webcam
python main.py --width 1920 --height 1080
python main.py --record               # Auto-start recording
```

---

## 🤚 Gesture Reference

| Gesture | Hand Pose | Action |
|---|---|---|
| ☝️ Index finger up | Only index extended | **Draw** |
| 🖐️ Open palm | All 5 fingers spread | **Stop / Idle** |
| ✌️ V-sign | Index + middle up | **Erase** |
| 🤙 Shaka | Thumb + pinky out | **Next color** |
| 👍 Thumbs up | Thumb up, fist | **Increase brush** |
| 👎 Thumbs down | Thumb down, fist | **Decrease brush** |
| ✊ Fist (hold 1.5s) | All fingers closed | **Clear canvas** |
| 👌 OK sign (hold 1s) | Thumb-index circle | **Save drawing** |
| 🤘 Index + pinky | Index + pinky up | **Undo** |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save canvas |
| `C` | Clear canvas |
| `U` | Undo |
| `R` | Toggle recording |

---

## 📁 Project Structure

```
├── main.py                  # Entry point
├── requirements.txt
├── README.md
├── config/
│   └── settings.py          # All constants & thresholds
├── core/
│   ├── hand_tracker.py      # MediaPipe wrapper
│   ├── gesture_engine.py    # Gesture classification
│   ├── state_manager.py     # FSM mode control
│   ├── smoothing.py         # Kalman filter & EMA
│   ├── drawing_engine.py    # Canvas & stroke management
│   ├── ui_renderer.py       # HUD overlay
│   └── export_manager.py    # Save & record
├── utils/
│   └── geometry.py          # Math helpers
├── tests/
│   └── test_gestures.py     # Unit tests
└── output/                  # Saved drawings (auto-created)
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 🔧 Configuration

All parameters are in `config/settings.py` — camera resolution, gesture thresholds, color palette, smoothing constants, UI sizes. Adjust without modifying any logic code.

---

## 📋 Technical Highlights

- **Kalman Filter** with constant-velocity model for predictive coordinate smoothing (low lag + low jitter)
- **Finite State Machine** with debounce (gesture must persist 4 frames) and confidence gating (≥65%)
- **Time-gated destructive actions** — fist-hold for clear and OK-hold for save prevent accidental triggers
- **Rising-edge one-shot actions** — color/size changes fire exactly once per gesture activation
- **Alpha-blended canvas** compositing for clean overlay on camera feed
- **Modular architecture** — each module has a single responsibility with a defined interface

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

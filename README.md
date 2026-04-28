<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.8.0-red.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0.10.0-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Status-Ultra%20Optimized-success.svg" alt="Status">
</div>

<h1 align="center">🎨 Air Canvas PRO</h1>

<p align="center">
  <b>A hyper-optimized, zero-latency Computer Vision application that lets you draw freely in the air using native hand gestures.</b>
</p>

---

## ⚡ Overview

Air Canvas PRO revolutionizes simple webcam tracking by utilizing a highly sophisticated **hybrid CPU/GPU architecture**. Running on Google MediaPipe and OpenCV, the application fundamentally eliminates IO boundaries and artificial frame drops. 

We completely decoupled traditional synchronous bottlenecks by spinning up multithreaded daemon processes and replacing legacy matrix operations with lightning-fast pure math. The result? A perfectly fluid, anti-aliased digital ink experience operating at maximum hardware capable FPS.

## 🧰 Tech Stack
* **Core Language:** Python 3.9+
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** Google MediaPipe (Hand Landmark Object Detection)
* **High-Performance Math:** NumPy
* **Hardware Acceleration:** OpenCL (T-API) & Windows PowerShell Interop
* **Concurrency Engine:** Python Native `threading`

## ⚙️ How It Works (The Engine)
1. **Asynchronous Daemon Threading:** The core application spawns a background `WebcamStream` process that exclusively handles natively pulling uncompressed 1080p frames from your camera. This ensures your computer's bandwidth never blocks the main logic cycle.
2. **AI Inference Downscaling:** The raw 1080p frame is cloned and aggressively downscaled to `640x360` before being handed to MediaPipe. This violently slashes the core processing load natively required by the XNNPACK Neural Logic, while retaining perfectly accurate spatial mapping for coordinates.
3. **Continuous Tracking & Interpolation:** The tip of the index finger (Landmark 8) is locked on. Between frames, a quadratic `C1` continuous **Bézier Spline engine** mathematically reconstructs the gaps in your real-world movement. The dynamic velocity-adaptive Euro-filter guarantees the ink snaps instantly when moving fast, but smooths hand-tremors out natively when slowing down.
4. **Numpy Memory Overlay:** Instead of calculating matrix bitwise logic (which brings most codebases to their knees at 1080p), the script projects the transparent ink canvas directly onto the camera feed using pure NumPy threshold indexing (`img[mask] = canvas[mask]`).

## ✨ Technical Marvels

* 🚀 **Zero-Latency Camera Threading:** By tearing down synchronous `cap.read()` wait times and isolating the hardware feed onto an asynchronous background Daemon thread, the main computational loop natively runs as fast as your processor allows.
* 🛠️ **Velocity-Adaptive Euro Filtering:** Eradicates the sluggish "trailing" sensation of typical drawing apps. The script actively calculates the physical kinetic velocity of your finger:
  * **Moving Fast:** Exponential smoothing is zeroed out instantly, snapping strictly to your coordinates with pristine zero-latency.
  * **Moving Slow/Hovering:** Heavy continuous interpolation dynamically kicks in, actively scrubbing out micro-jitter and hand tremors from the video feed.
* 🧮 **Numpy Boolean Mask Compositing:** Bypasses sluggish `cv2.bitwise` logic. By using an ultra-fast underlying NumPy memory assignment, compiling your 1080p canvas onto the live feed uses negligible overhead.
* 🎭 **Quadratic Bézier Spline Engine:** Digital-ink rendering achieved by generating mathematically perfect `C1` continuous quadratic splines with stamped radial caps, meaning your handwriting will look fluidly gorgeous without choppy triangle edges.
* 🔥 **T-API GPU Hooks:** Dynamically detects your active dedicated NVIDIA or AMD chip using native OS telemetry and strictly routes intensive scaling permutations to GPU OpenCL nodes.

## 👋 Intuitive Gestures

| Gesture | Mode | Description |
| :--- | :--- | :--- |
| **☝️ Index Finger Extended** | `DRAW` | Extrapolates movement into beautiful continuous strokes on the canvas. |
| **🤏 Pinch (Index & Thumb)** | `MOVE` | Pinches the coordinate system, allowing you to physically drag your entire drawing across space. |
| **✊ Closed Fist** | `ERASE` | Exerts a radius of annihilation, cleaning precise spots like a digital real-world eraser. |
| **✋ Open Palm** | `IDLE` | Disengages writing entirely; acts as a non-destructive hovering cursor. |

### 🎹 Hotkeys
* `c` : Hard resets the canvas board immediately.
* `ESC` : Safely terminates background threads, flushes cache, and powers off the camera matrix gracefully.

## 🚀 Getting Started

### Prerequisites
Make sure your system has Python setup with appropriate camera driver connectivity.

### Installation
1. Clone the repository natively:
```bash
git clone https://github.com/RaidenX2905/Air-Canvas.git
cd Air-Canvas
```

2. Install the core dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
python air_canvas.py
```

---

## 🛠️ Advanced: Virtual Environment Setup
If you prefer to keep your global environment clean, you can also run this in a Virtual Environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python air_canvas.py
```

> **Note:** Upon successful execution, the terminal will instantly output a line confirming whether a Dedicated GPU, Integrated GPU, or CPU Fallback is actively being used to run the background engine.

<p align="center">
  <i>Engineered for next-level fluidity.</i>
</p>

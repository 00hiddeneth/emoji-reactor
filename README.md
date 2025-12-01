за # 🎬 Rocky Gesture Recognition System v2.0

A real-time camera-based gesture recognition system using advanced computer vision techniques to detect facial expressions and hand gestures, displaying corresponding Rocky-themed images.

## ✨ Features

### **Advanced Pattern Detection**
- **🔫 Pat_2: Finger-Gun** - Hand geometry analysis (finger positions + temple proximity) [HIGHEST PRIORITY]
- **👅 Pat_4: Tongue Out** - HSV color space analysis for red/pink pixels in mouth region
- **🕶️ Pat_3: Sunglasses** - Haar Cascade eye detection (occlusion-based)
- **👁️ Pat_1: One Eye Closed** - Eye Aspect Ratio (EAR) for precise wink detection

### **Smart Technology**
- **Temporal Smoothing**: 5-frame majority vote eliminates flickering
- **Priority System**: Finger-gun > Tongue > Sunglasses > One eye
- **Haar Cascade**: Advanced eye occlusion detection for sunglasses
- **Debug Mode**: Live ROI overlays and metric visualization (press 'd')
- **Configurable**: All thresholds in centralized CONFIG dictionary
- **High-Quality Output**: 600x600 Rocky images with LANCZOS4 interpolation

## Requirements

- Python 3.12 (Homebrew: `brew install python@3.12`)
- macOS or Windows with a webcam
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (Python 3.12) and install deps:**
   ```bash
   # macOS: ensure Python 3.12 is installed
   brew install python@3.12

   # Create and activate a virtual environment
   python3.12 -m venv emoji_env
   source emoji_env/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Ensure you have the Rocky images in the project directory:**
   - `Rocky-clip.JPG` - Squinting eye + pointing finger
   - `Rocky-gun.JPG` - Hand gun gesture
   - `Rocky-sunglasses.JPG` - Wearing sunglasses
   - `Rocky-tongue.JPG` - Tongue out

## Usage

1. **Run the application:**
   ```bash
   # Option A: use helper script
   ./run.sh

   # Option B: run manually
   source emoji_env/bin/activate
   python3 emoji_reactor.py
   ```

2. **Two windows will open:**
   - **Camera Feed**: Shows your live camera with detection status
   - **Rocky Output**: Displays the corresponding Rocky image based on your gestures

3. **Controls:**
   - Press `q` to quit the application
   - Try different gestures to see Rocky reactions!

## Gestures & Triggers

### 🔫 Gun Gesture (Highest Priority)
- Raise hand near your head (temple area)
- Hand should be above shoulder level
- Hand positioned near ear

### 🕶️ Sunglasses
- Wear sunglasses or create dark areas over eyes
- Detection based on pixel intensity around eye regions

### 👁️👉 Squint + Pointing Finger
- Squint one eye (not fully closed)
- Point index finger towards your body
- Both conditions must be met simultaneously

### 👅 Tongue Out
- Stick your tongue out
- Detection based on lip-to-chin distance ratio
- Threshold: 0.025

## How It Works

The application uses MediaPipe solutions and OpenCV Haar Cascade:

1. **MediaPipe FaceMesh**: 478 facial landmarks for eye and mouth analysis
2. **MediaPipe Hands**: 21 hand landmarks for finger-gun gesture detection
3. **Haar Cascade Classifier**: Eye detection for sunglasses occlusion
4. **HSV Color Analysis**: Tongue detection via red/pink pixel ratio
5. **Temporal Smoothing**: Majority vote across 5 frames for stability

### Detection Priority
1. **🔫 Finger-Gun Gesture** (highest) - Overrides all other detections
2. **👅 Tongue Out** - Second priority
3. **🕶️ Sunglasses** - Third priority
4. **👁️ One Eye Closed** - Fourth priority (baseline trigger)
5. **⚪ Neutral** - Default when no gesture detected (black screen)

## Customization

### Adjusting Detection Sensitivity

Edit threshold values in `emoji_reactor.py`:

```python
TONGUE_THRESHOLD = 0.025  # Increase for less sensitive tongue detection
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Adjust for squint sensitivity
```

### Changing Rocky Images
Replace the image files with your own:
- `Rocky-clip.JPG` - Squint + pointing gesture
- `Rocky-gun.JPG` - Gun gesture
- `Rocky-sunglasses.JPG` - Sunglasses
- `Rocky-tongue.JPG` - Tongue out

**Note**: File names are case-sensitive (.JPG extension required)

## Troubleshooting

### Camera Issues (macOS)
- If you see "not authorized to capture video", grant Camera access for your terminal/editor:
  - System Settings → Privacy & Security → Camera → enable for Terminal/VS Code/iTerm
- Quit and relaunch the terminal/editor after changing permissions
- Ensure no other app is using the camera
- Try different camera indices by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Rocky Images Not Loading
- Verify image files are in the same directory as the script
- Check file names match exactly: `Rocky-clip.JPG`, `Rocky-gun.JPG`, `Rocky-sunglasses.JPG`, `Rocky-tongue.JPG`
- Ensure image files are not corrupted
- **Note**: File extensions are case-sensitive (.JPG not .jpg)

### Detection Issues
- **Gun gesture**: Raise hand higher, closer to head/temple area
- **Sunglasses**: Ensure good contrast, try actual sunglasses
- **Squint**: Don't fully close eye, just squint one eye
- **Pointing**: Index finger should point clearly toward your body center
- **Tongue**: Stick tongue out more prominently
- Ensure good lighting on your face
- Keep your face and hands clearly visible in the camera

## Technical Details

- **OpenCV 4.12.0**: Camera capture, image display, Haar Cascade
- **MediaPipe 0.10.14**: FaceMesh (478 landmarks), Hands (21 landmarks)
- **NumPy 2.2.6**: Mathematical operations and array processing
- **Haar Cascade**: haarcascade_eye.xml for sunglasses detection
- **Image Output**: 600x600 pixels, 1:1 aspect ratio, LANCZOS4 interpolation
- **Real-time processing**: 30+ FPS on modern hardware

## Dependencies

- `opencv-python==4.12.0.88` - Core computer vision library
- `opencv-contrib-python==4.12.0.88` - Extended modules (required by MediaPipe)
- `mediapipe==0.10.14` - Multi-modal ML solutions (FaceMesh, Hands)
- `numpy==2.2.6` - Numerical computing and array operations

**Note**: Both `opencv-python` and `opencv-contrib-python` are required. Do not uninstall `opencv-contrib-python`.

See `requirements.txt` for installation and `requirements-lock.txt` for pinned versions.

## License

MIT License - see LICENSE file for details.

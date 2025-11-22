# How to View Your Humanoid Videos

## Problem Summary
Your system has OpenGL/GLX issues that prevent direct video playback and rendering. This is common in remote sessions, WSL, or systems with misconfigured graphics drivers.

## ✅ Solution: View Frames in Web Browser

I've created HTML galleries that allow you to view the video frames in any web browser, completely bypassing the OpenGL issue.

### Available Galleries

1. **100M Model** (Latest - 21-33-51):
   - Gallery: `video_gallery_100M.html`
   - Frames: `frames_100M/` directory (30 frames)
   - Performance: Variable (4603 ± 3743 reward)

2. **50M Model** (17-47-25):
   - Gallery: `video_gallery_50M.html`
   - Frames: `frames_50M/` directory (30 frames)
   - Performance: Excellent (9776 ± 91 reward)

### How to Open

```bash
# Method 1: Firefox (should already be opening)
firefox video_gallery_100M.html
firefox video_gallery_50M.html

# Method 2: Google Chrome
google-chrome video_gallery_100M.html

# Method 3: Default browser
xdg-open video_gallery_100M.html
```

### Gallery Features

The HTML galleries provide:
- **Gallery Mode**: See all frames at once in a grid
- **Slideshow Mode**: View frames one at a time
- **Play/Pause**: Animate through frames
- **Speed Control**: Adjust playback speed
- **Keyboard Controls**:
  - Arrow Right / Spacebar: Next frame
  - Arrow Left: Previous frame
  - `P`: Play/Pause

## 📹 Original Videos

The full MP4 videos are still available in case you can view them on another machine or transfer them:

### 100M Model Videos
- `videos/eval-Humanoid-v5-step-0-to-step-1000.mp4` (1.1MB) - Episode 1: Perfect run (9750 reward)
- `videos/eval-Humanoid-v5-step-1370-to-step-2370.mp4` (304KB) - Episodes 2-3: Falls

### 50M Model Videos
- `videos/50M/eval-Humanoid-v5-step-0-to-step-1000.mp4` (1.1MB) - Episode 1
- `videos/50M/eval-Humanoid-v5-step-1000-to-step-2000.mp4` (1.1MB) - Episode 2
- `videos/50M/eval-Humanoid-v5-step-2000-to-step-3000.mp4` (1.1MB) - Episode 3

## 🔧 Generate More Videos/Frames

### Create new evaluation videos:

```bash
# 100M model - 10 episodes
./run_2d_display.sh --use-final-model --env-id Humanoid-v5 --episodes 10

# 50M model - 10 episodes  
./run_2d_display.sh --use-final-model --env-id Humanoid-v5 \\
  --run outputs/2025-10-28/17-47-25 \\
  --video-dir ./videos/50M --episodes 10
```

### Extract frames from any video:

```bash
# Extract 5 frames per second (default)
./extract_frames.sh videos/eval-Humanoid-v5-step-0-to-step-1000.mp4

# Extract 10 frames per second for smoother animation
./extract_frames.sh videos/eval-Humanoid-v5-step-0-to-step-1000.mp4 my_frames 10
```

### Create HTML gallery from frames:

```bash
python3 create_video_gallery.py frames_100M my_gallery.html
```

## 📊 Performance Comparison

| Model | Training Steps | Mean Reward | Std Dev | Consistency |
|-------|---------------|-------------|---------|-------------|
| 50M   | 50,000,000    | 9776.07     | ±91.91  | ⭐⭐⭐⭐⭐ Excellent |
| 100M  | 100,000,000   | 4603.77     | ±3743.43| ⭐⭐ Poor   |

**Recommendation**: The 50M model is significantly better. Use it for deployment.

## 🐛 Alternative: Fix OpenGL (Advanced)

If you want to fix the OpenGL issue for other applications:

```bash
# Check OpenGL info
glxinfo | grep "OpenGL"

# Install missing drivers (NVIDIA example)
sudo apt-get install nvidia-driver-XXX

# Or use software rendering (slower)
export LIBGL_ALWAYS_SOFTWARE=1
```

## 💡 Tips

1. **Transfer videos**: Copy MP4 files to your local machine to view with any video player
2. **Share results**: The HTML galleries can be opened on any computer with a browser
3. **No OpenGL needed**: All frame viewing solutions work without any graphics drivers

## Need Help?

- Frames not showing? Check that the paths in the HTML file are correct
- Browser not opening? Try: `DISPLAY=:0 firefox video_gallery_100M.html`
- Want different frame rate? Re-run `extract_frames.sh` with different FPS parameter

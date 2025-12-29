#!/usr/bin/env bash
# Extract frames from video files as PNG images
# Usage: ./extract_frames.sh <video_file> [output_dir]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <video_file> [output_dir] [frame_rate]"
  echo "Example: $0 videos/eval-Humanoid-v5-step-0-to-step-1000.mp4"
  echo "         $0 videos/eval-Humanoid-v5-step-0-to-step-1000.mp4 frames_output 2"
  exit 1
fi

VIDEO_FILE="$1"
VIDEO_BASENAME=$(basename "$VIDEO_FILE" .mp4)
OUTPUT_DIR="${2:-frames_${VIDEO_BASENAME}}"
FRAME_RATE="${3:-5}"  # Extract 5 frames per second by default

if [[ ! -f "$VIDEO_FILE" ]]; then
  echo "ERROR: Video file not found: $VIDEO_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Extracting frames from: $VIDEO_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Frame rate: $FRAME_RATE fps"
echo ""

# Use ffmpeg to extract frames
ffmpeg -i "$VIDEO_FILE" \
  -vf "fps=$FRAME_RATE" \
  "$OUTPUT_DIR/frame_%04d.png" \
  -y 2>&1 | grep -E "(frame=|Duration:)" || true

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR"/frame_*.png 2>/dev/null | wc -l)

echo ""
echo "✓ Extracted $FRAME_COUNT frames to $OUTPUT_DIR/"
echo ""
echo "You can view the frames with an image viewer:"
echo "  eog $OUTPUT_DIR/frame_0001.png  # Eye of GNOME"
echo "  xdg-open $OUTPUT_DIR/frame_0001.png"
echo ""
echo "Or create a contact sheet (grid of frames):"
echo "  montage $OUTPUT_DIR/frame_*.png -tile 4x4 -geometry +2+2 $OUTPUT_DIR/contact_sheet.png"

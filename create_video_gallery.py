#!/usr/bin/env python3
"""
Create an HTML gallery to view extracted video frames in a web browser.
This works around OpenGL/display issues by viewing frames in a browser.
"""

import os
import sys
from pathlib import Path

def create_html_gallery(frames_dir, output_html="gallery.html"):
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        print(f"ERROR: Directory not found: {frames_dir}")
        sys.exit(1)
    
    # Find all frame images
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        print(f"ERROR: No frame_*.png files found in {frames_dir}")
        sys.exit(1)
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Video Frames Gallery - {frames_dir.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1e1e1e;
            color: #ffffff;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            background-color: #1e1e1e;
            padding: 10px;
            z-index: 100;
        }}
        button {{
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            cursor: pointer;
            background-color: #0066cc;
            color: white;
            border: none;
            border-radius: 5px;
        }}
        button:hover {{
            background-color: #0052a3;
        }}
        .slideshow-container {{
            position: relative;
            max-width: 800px;
            margin: auto;
            background-color: #000;
        }}
        .slideshow-image {{
            width: 100%;
            height: auto;
        }}
        .frame-info {{
            text-align: center;
            padding: 10px;
            font-size: 18px;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            padding: 20px;
        }}
        .gallery-item {{
            cursor: pointer;
            border: 2px solid #333;
            transition: transform 0.2s;
        }}
        .gallery-item:hover {{
            transform: scale(1.05);
            border-color: #0066cc;
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <h1>{frames_dir.name.replace('_', ' ').title()} - {len(frame_files)} Frames</h1>
    
    <div class="controls">
        <button onclick="showSlideshow()">📺 Slideshow Mode</button>
        <button onclick="showGallery()">🖼️ Gallery Mode</button>
        <button id="playBtn" onclick="togglePlay()" class="hidden">▶️ Play</button>
        <button onclick="prevFrame()" class="hidden" id="prevBtn">⏮️ Previous</button>
        <button onclick="nextFrame()" class="hidden" id="nextBtn">⏭️ Next</button>
        <label for="speed" class="hidden" id="speedLabel">Speed: 
            <select id="speed" onchange="updateSpeed()">
                <option value="2000">Slow (2s)</option>
                <option value="500" selected>Normal (0.5s)</option>
                <option value="100">Fast (0.1s)</option>
            </select>
        </label>
    </div>
    
    <div id="slideshow" class="slideshow-container hidden">
        <img id="slideshowImg" class="slideshow-image" src="">
        <div class="frame-info" id="frameInfo"></div>
    </div>
    
    <div id="gallery" class="gallery-grid">
"""
    
    # Add all frames to gallery
    for i, frame_file in enumerate(frame_files, 1):
        rel_path = frame_file.relative_to(frames_dir.parent)
        html_content += f"""        <div class="gallery-item" onclick="showFrame({i-1})">
            <img src="{rel_path}" alt="Frame {i}">
        </div>
"""
    
    html_content += """    </div>
    
    <script>
        const frames = [
"""
    
    # Add frame paths to JavaScript
    for frame_file in frame_files:
        rel_path = frame_file.relative_to(frames_dir.parent)
        html_content += f'            "{rel_path}",\n'
    
    html_content += """        ];
        
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;
        let speed = 500;
        
        function showSlideshow() {
            document.getElementById('slideshow').classList.remove('hidden');
            document.getElementById('gallery').classList.add('hidden');
            document.getElementById('playBtn').classList.remove('hidden');
            document.getElementById('prevBtn').classList.remove('hidden');
            document.getElementById('nextBtn').classList.remove('hidden');
            document.getElementById('speedLabel').classList.remove('hidden');
            showFrame(currentFrame);
        }
        
        function showGallery() {
            document.getElementById('slideshow').classList.add('hidden');
            document.getElementById('gallery').classList.remove('hidden');
            document.getElementById('playBtn').classList.add('hidden');
            document.getElementById('prevBtn').classList.add('hidden');
            document.getElementById('nextBtn').classList.add('hidden');
            document.getElementById('speedLabel').classList.add('hidden');
            if (playing) togglePlay();
        }
        
        function showFrame(index) {
            if (index < 0) index = frames.length - 1;
            if (index >= frames.length) index = 0;
            currentFrame = index;
            document.getElementById('slideshowImg').src = frames[currentFrame];
            document.getElementById('frameInfo').textContent = `Frame ${currentFrame + 1} / ${frames.length}`;
            showSlideshow();
        }
        
        function nextFrame() {
            showFrame(currentFrame + 1);
        }
        
        function prevFrame() {
            showFrame(currentFrame - 1);
        }
        
        function togglePlay() {
            playing = !playing;
            document.getElementById('playBtn').textContent = playing ? '⏸️ Pause' : '▶️ Play';
            
            if (playing) {
                playInterval = setInterval(nextFrame, speed);
            } else {
                if (playInterval) clearInterval(playInterval);
            }
        }
        
        function updateSpeed() {
            speed = parseInt(document.getElementById('speed').value);
            if (playing) {
                if (playInterval) clearInterval(playInterval);
                playInterval = setInterval(nextFrame, speed);
            }
        }
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            if (document.getElementById('slideshow').classList.contains('hidden')) return;
            
            if (e.key === 'ArrowRight' || e.key === ' ') {
                nextFrame();
                e.preventDefault();
            } else if (e.key === 'ArrowLeft') {
                prevFrame();
                e.preventDefault();
            } else if (e.key === 'p' || e.key === 'P') {
                togglePlay();
                e.preventDefault();
            }
        });
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path = frames_dir.parent / output_html
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created HTML gallery: {output_path}")
    print(f"  Frames: {len(frame_files)}")
    print(f"\nTo view:")
    print(f"  firefox {output_path}")
    print(f"  google-chrome {output_path}")
    print(f"  xdg-open {output_path}")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_video_gallery.py <frames_directory> [output.html]")
        print("Example: python3 create_video_gallery.py frames_100M")
        sys.exit(1)
    
    frames_dir = sys.argv[1]
    output_html = sys.argv[2] if len(sys.argv) > 2 else "video_gallery.html"
    
    create_html_gallery(frames_dir, output_html)

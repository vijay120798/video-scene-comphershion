# Offline Video Scene Comprehension

A complete video analysis solution that works **without any API keys** or external dependencies.

## Features

- 🎬 **Frame Extraction**: Extracts frames from videos at configurable intervals
- 🔍 **Object Detection**: Detects objects using computer vision techniques
- 📝 **Scene Analysis**: Generates descriptions of what's happening in each frame
- 📊 **Video Summary**: Provides overall video comprehension
- 💾 **Export Options**: Download results as JSON or text files

## Installation

```bash
pip install -r requirements_offline.txt
```

## Usage

```bash
streamlit run offline_video_analyzer.py
```

## How It Works

### No API Keys Required!

This solution uses:
- **OpenCV** for computer vision processing
- **NumPy** for numerical computations
- **Color Analysis** to detect objects, people, and scenes
- **Edge Detection** for shape analysis
- **HSV Color Space** for better object recognition

### Detection Capabilities

1. **People Detection**: Uses skin tone detection
2. **Object Detection**: Edge and contour analysis
3. **Scene Classification**: Indoor/outdoor detection
4. **Lighting Analysis**: Brightness and contrast measurement
5. **Vegetation Detection**: Green color analysis

## Output Format

The system provides:
- **Frame-by-frame analysis** with timestamps
- **Object detection results** with confidence scores
- **Action inference** based on detected objects
- **Complete video summary** in natural language
- **Exportable results** in JSON and text formats

## Advantages

✅ **No API costs** - completely free
✅ **No rate limits** - process unlimited videos
✅ **Privacy** - all processing happens locally
✅ **Fast** - no network latency
✅ **Reliable** - no API downtime issues

## Limitations

- Less sophisticated than cloud AI models
- Limited to visual object detection
- No advanced semantic understanding
- Accuracy depends on video quality

## Configuration

You can modify these parameters in the code:
- `interval_sec`: Frame extraction interval (default: 2.0 seconds)
- `max_frames`: Maximum frames to analyze (default: 8)
- Detection thresholds for various objects

## Example Output

```
Video Analysis Summary:
- Total frames analyzed: 5
- Duration: 10.0 seconds
- Objects detected: person, object, light_source
- Activities detected: movement_detected, illumination

Scene by scene breakdown:
Time 0.0s: Scene contains people with objects: person, object. Actions: movement_detected, illumination. Scene appears bright.
Time 2.0s: Indoor scene with lighting. Objects: object, light_source. Scene has normal lighting.
...
```

Run this solution and upload any video to get instant scene comprehension without any API keys!

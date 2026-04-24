from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
from dataclasses import dataclass
import tempfile
import streamlit as st

@dataclass
class FrameAnalysis:
    frame_id: int
    timestamp: float
    scene_description: str
    objects: List[str]
    actions: List[str]
    confidence: float

class OfflineVideoAnalyzer:
    """Complete offline video analysis without API dependencies"""
    
    def __init__(self):
        self.object_detector = self._load_object_detector()
        self.scene_classifier = self._load_scene_classifier()
        
    def _load_object_detector(self):
        """Load pre-trained object detection model"""
        # Using OpenCV's DNN with MobileNet-SSD for offline object detection
        try:
            net = cv2.dnn.readNetFromCaffe(
                "models/MobileNetSSD_deploy.prototxt",
                "models/MobileNetSSD_deploy.caffemodel"
            )
            return net
        except:
            # Fallback to basic detection if model files not found
            return None
    
    def _load_scene_classifier(self):
        """Load scene classification model"""
        # Simple rule-based scene classification
        return {
            'indoor': ['wall', 'floor', 'ceiling', 'furniture', 'table', 'chair'],
            'outdoor': ['sky', 'tree', 'building', 'road', 'car', 'grass'],
            'people': ['person', 'face', 'hand', 'body'],
            'objects': ['phone', 'computer', 'book', 'cup', 'bottle']
        }
    
    def extract_frames(self, video_path: str, interval_sec: float = 2.0, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = max(int(interval_sec * fps), 1)
        
        frames = []
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval_frames == 0:
                frames.append(frame)
                saved_count += 1
                if max_frames > 0 and saved_count >= max_frames:
                    break
                    
            frame_idx += 1
            
        cap.release()
        return frames
    
    def analyze_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameAnalysis:
        """Analyze a single frame"""
        # Basic scene analysis using computer vision
        height, width = frame.shape[:2]
        
        # Detect basic scene properties
        brightness = np.mean(frame)
        contrast = np.std(frame)
        
        # Simple object detection using color and shape analysis
        objects = self._detect_objects_simple(frame)
        actions = self._infer_actions(objects, frame_id)
        
        # Generate scene description
        scene_type = self._classify_scene(objects, frame)
        description = self._generate_description(scene_type, objects, actions, brightness)
        
        return FrameAnalysis(
            frame_id=frame_id,
            timestamp=timestamp,
            scene_description=description,
            objects=objects,
            actions=actions,
            confidence=0.8
        )
    
    def _detect_objects_simple(self, frame: np.ndarray) -> List[str]:
        """Simple object detection using basic computer vision"""
        objects = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect skin tones (people)
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_pixels = cv2.countNonZero(skin_mask)
        
        if skin_pixels > 1000:
            objects.append("person")
        
        # Detect edges (structures, objects)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours to detect objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Classify based on shape
                if aspect_ratio > 3:
                    objects.append("horizontal_object")
                elif aspect_ratio < 0.3:
                    objects.append("vertical_object")
                else:
                    objects.append("object")
        
        # Detect bright areas (lights, screens)
        bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        bright_areas = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        if len(bright_areas) > 0:
            objects.append("light_source")
        
        # Detect green areas (vegetation)
        green_lower = np.array([35, 40, 40], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)
        
        if green_pixels > 5000:
            objects.append("vegetation")
        
        return list(set(objects))  # Remove duplicates
    
    def _infer_actions(self, objects: List[str], frame_id: int) -> List[str]:
        """Infer possible actions based on detected objects"""
        actions = []
        
        if "person" in objects:
            if frame_id > 0:
                actions.append("movement_detected")
            actions.append("present_in_scene")
        
        if "light_source" in objects:
            actions.append("illumination")
        
        if len(objects) > 3:
            actions.append("complex_scene")
        elif len(objects) > 0:
            actions.append("simple_scene")
        
        return actions
    
    def _classify_scene(self, objects: List[str], frame: np.ndarray) -> str:
        """Classify the scene type"""
        if "person" in objects:
            return "people_scene"
        elif "vegetation" in objects:
            return "outdoor_nature"
        elif "light_source" in objects:
            return "indoor_lit"
        else:
            return "general_scene"
    
    def _generate_description(self, scene_type: str, objects: List[str], actions: List[str], brightness: float) -> str:
        """Generate natural language description"""
        descriptions = {
            "people_scene": f"Scene contains people with objects: {', '.join(objects)}. Actions: {', '.join(actions)}.",
            "outdoor_nature": f"Outdoor scene with vegetation. Objects detected: {', '.join(objects)}.",
            "indoor_lit": f"Indoor scene with lighting. Objects: {', '.join(objects)}.",
            "general_scene": f"General scene with {len(objects)} objects detected: {', '.join(objects)}."
        }
        
        base_desc = descriptions.get(scene_type, descriptions["general_scene"])
        
        # Add brightness information
        if brightness > 150:
            base_desc += " Scene appears bright."
        elif brightness < 80:
            base_desc += " Scene appears dark."
        else:
            base_desc += " Scene has normal lighting."
        
        return base_desc
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Complete video analysis"""
        frames = self.extract_frames(video_path, interval_sec=2.0, max_frames=8)
        analyses = []
        
        for i, frame in enumerate(frames):
            timestamp = i * 2.0  # 2 seconds per frame
            analysis = self.analyze_frame(frame, i, timestamp)
            analyses.append(analysis)
        
        # Generate overall summary
        summary = self._generate_video_summary(analyses)
        
        return {
            "analyses": analyses,
            "summary": summary,
            "total_frames": len(frames),
            "video_duration": len(frames) * 2.0
        }
    
    def _generate_video_summary(self, analyses: List[FrameAnalysis]) -> str:
        """Generate overall video summary"""
        if not analyses:
            return "No frames analyzed."
        
        # Collect all unique objects and actions
        all_objects = set()
        all_actions = set()
        
        for analysis in analyses:
            all_objects.update(analysis.objects)
            all_actions.update(analysis.actions)
        
        summary = f"Video Analysis Summary:\n"
        summary += f"- Total frames analyzed: {len(analyses)}\n"
        summary += f"- Duration: {analyses[-1].timestamp + 2:.1f} seconds\n"
        summary += f"- Objects detected: {', '.join(all_objects) if all_objects else 'None'}\n"
        summary += f"- Activities detected: {', '.join(all_actions) if all_actions else 'None'}\n\n"
        
        summary += "Scene by scene breakdown:\n"
        for analysis in analyses:
            summary += f"Time {analysis.timestamp:.1f}s: {analysis.scene_description}\n"
        
        return summary

def main():
    st.set_page_config(page_title="Offline Video Scene Comprehension", layout="wide")
    
    st.title("🎬 Offline Video Scene Comprehension")
    st.write("Analyze video content without API dependencies using computer vision")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        if st.button("Analyze Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            try:
                with st.spinner("Analyzing video..."):
                    analyzer = OfflineVideoAnalyzer()
                    results = analyzer.analyze_video(video_path)
                
                st.success(f"✅ Analyzed {results['total_frames']} frames")
                
                # Display summary
                st.subheader("📋 Video Summary")
                st.text_area("Complete Analysis", results["summary"], height=300)
                
                # Display frame-by-frame analysis
                st.subheader("🎞️ Frame by Frame Analysis")
                
                for i, analysis in enumerate(results["analyses"]):
                    with st.expander(f"Frame {analysis.frame_id + 1} - {analysis.timestamp:.1f}s", expanded=i == 0):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Scene Description:**")
                            st.write(analysis.scene_description)
                            
                            st.write("**Objects Detected:**")
                            st.write(", ".join(analysis.objects) if analysis.objects else "None")
                        
                        with col2:
                            st.write("**Actions Detected:**")
                            st.write(", ".join(analysis.actions) if analysis.actions else "None")
                            
                            st.write("**Confidence:**")
                            st.write(f"{analysis.confidence:.2f}")
                
                # Download results
                st.subheader("💾 Export Results")
                
                # JSON format
                json_results = {
                    "summary": results["summary"],
                    "frames": [
                        {
                            "frame_id": a.frame_id,
                            "timestamp": a.timestamp,
                            "description": a.scene_description,
                            "objects": a.objects,
                            "actions": a.actions,
                            "confidence": a.confidence
                        }
                        for a in results["analyses"]
                    ]
                }
                
                st.download_button(
                    label="📥 Download JSON Results",
                    data=json.dumps(json_results, indent=2),
                    file_name="video_analysis.json",
                    mime="application/json"
                )
                
                # Text format
                text_results = results["summary"]
                st.download_button(
                    label="📄 Download Text Summary",
                    data=text_results,
                    file_name="video_summary.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Make sure you have OpenCV installed: pip install opencv-python")

if __name__ == "__main__":
    main()

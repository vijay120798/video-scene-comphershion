from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.visual_rag.rag_pipeline import VisualRAGPipeline

st.set_page_config(page_title="Video Visual RAG", layout="wide")


@st.cache_resource
def get_pipeline() -> "VisualRAGPipeline":
    from src.visual_rag.config import settings
    from src.visual_rag.rag_pipeline import VisualRAGPipeline
    return VisualRAGPipeline(settings)


def _ensure_pipeline():
    """Get pipeline or show error and return None."""
    try:
        return get_pipeline()
    except Exception as e:
        st.error(f"Pipeline failed to load: {e}")
        st.info(
            "Check that `.env` has valid OPENAI_API_KEY, GEMINI_API_KEY, and DEVICE=cpu if you have no GPU."
        )
        return None


st.title("Video Understanding with Visual-Only RAG")
st.write(
    "Upload a video, then ask a question about what happens in the video. "
    "This app uses a visual-only RAG pipeline (no audio)."
)

uploaded_video = st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi", "mkv"]
)
question = st.text_input("Ask a question about the video")

pipeline = _ensure_pipeline()
if pipeline is None:
    st.stop()

if uploaded_video is not None:
    if st.button("Index video"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_video.name).suffix
        ) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        try:
            with st.spinner("Indexing video... this may take a while."):
                artifacts = pipeline.index_video(video_path)
            st.success(
                f"Indexed {len(artifacts.frames)} frames into {len(artifacts.chunks)} chunks."
            )
            st.session_state["video_path"] = video_path
            st.session_state["artifacts"] = artifacts
            
            # Display frame-level text conversion only
            st.subheader("📝 Frame Descriptions")
            
            # Show frame-by-frame analysis
            with st.expander("🎬 Frame-by-Frame Analysis", expanded=True):
                for frame in artifacts.frames:
                    st.write(f"**Frame {frame.frame_id}** at {frame.timestamp_sec:.2f}s")
                    st.text_area(
                        f"Frame {frame.frame_id} Details",
                        value=frame.merged_text,
                        height=200,
                        key=f"frame_{frame.frame_id}"
                    )
                    st.divider()
            
            # Export options (frame-level)
            st.subheader("💾 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Full Text"):
                    full_text = "VIDEO TO TEXT CONVERSION\n" + "="*50 + "\n\n"
                    full_text += f"Total Frames: {len(artifacts.frames)}\n"
                    full_text += f"Total Chunks: {len(artifacts.chunks)}\n\n"
                    
                    full_text += "FRAME-BY-FRAME ANALYSIS:\n" + "-"*30 + "\n"
                    for frame in artifacts.frames:
                        full_text += f"\nFrame {frame.frame_id} ({frame.timestamp_sec:.2f}s):\n"
                        full_text += frame.merged_text + "\n"
                    
                    st.download_button(
                        label="📥 Download Full Text",
                        data=full_text,
                        file_name="video_to_text_conversion.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📋 Export Frame Descriptions"):
                    summary_text = "FRAME DESCRIPTIONS\n" + "="*30 + "\n\n"
                    for frame in artifacts.frames:
                        summary_text += f"Frame {frame.frame_id} ({frame.timestamp_sec:.2f}s):\n"
                        summary_text += frame.merged_text + "\n\n"
                    
                    st.download_button(
                        label="📥 Download Frame Descriptions",
                        data=summary_text,
                        file_name="frame_descriptions.txt",
                        mime="text/plain"
                    )
                    
        except Exception as e:
            st.error(f"Indexing failed: {e}")


if "video_path" in st.session_state and question:
    if st.button("Get answer"):
        try:
            with st.spinner("Answering..."):
                result = pipeline.answer_question(question)
            st.subheader("Answer")
            st.write(result.get("answer", result))
            st.subheader("Retrieved context")
            st.json(result.get("context", []))
        except Exception as e:
            st.error(f"Answer failed: {e}")


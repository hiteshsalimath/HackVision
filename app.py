import streamlit as st
import os
from pathlib import Path
from anpr import run

# Set up Streamlit page configuration
st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="wide"
)

# App title
st.title("Automated Number Plate Recognition (ANPR) System")

# File upload
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded file to a temporary location
    temp_dir = Path("temp_videos")
    temp_dir.mkdir(exist_ok=True)
    input_video_path = temp_dir / uploaded_file.name

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded file saved to `{input_video_path}`.")

    # Generate output file name
    output_dir = Path("output_videos")
    output_dir.mkdir(exist_ok=True)
    file_name = Path(uploaded_file.name).stem  # Extract file name without extension
    output_path = output_dir / f"processed_{file_name}.mp4"

    # Run ANPR detection
    st.write("Processing video, please wait...")
    try:
        run(
            source=str(input_video_path),
            project=str(output_dir),
            name=f"processed_{file_name}",
            exist_ok=True,
            save_txt=True,  # Save text output
            nosave=False,  # Ensure the processed video is saved
        )

        st.success("Video processing complete!")
        st.video(str(output_path))

        # Provide a download link for the processed video
        st.write(f"Download the processed video: [Download Here](./{output_path})")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    finally:
        # Optional cleanup logic if needed
        st.info("Processing complete. Temporary files saved for review.")
else:
    st.info("Please upload a video file to start.")

# Footer
st.markdown("---")
st.markdown("Developed by [Hitesh](https://github.com/username)")  

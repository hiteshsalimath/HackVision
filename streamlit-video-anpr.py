import streamlit as st
import subprocess
import os

# Assuming HOME is an environment variable or defined path
HOME = os.getenv('HOME', '')  # Fallback to empty string if not set

def run_anpr_detection(confidence=0.1, device=0, weights=None, source=None):
    """
    Run ANPR detection using a subprocess call
    
    Args:
        confidence (float): Detection confidence threshold
        device (int): Device number for processing
        weights (str): Path to model weights
        source (str): Video source file
    """
    if not weights:
        weights = f"{HOME}/yolov9/runs/train/exp/weights/best.pt"
    
    command = [
        "python", "anpr.py", 
        "--conf", str(confidence),
        "--device", str(device),
        "--weights", weights,
        "--source", source
    ]
    
    try:
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Display output
        st.text("ANPR Detection Output:")
        st.text(result.stdout)
        
        # Check for any errors
        if result.stderr:
            st.error("Errors occurred during processing:")
            st.text(result.stderr)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.title("ANPR Video Detection")
    
    # Confidence threshold input
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.1)
    
    # Device selection
    device = st.number_input("Device Number", min_value=0, value=0)
    
    # Weights path input
    weights = st.text_input(
        "Model Weights Path", 
        value=f"{HOME}/yolov9/runs/train/exp/weights/best.pt"
    )
    
    # Video source upload
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Run detection button
        if st.button("Run ANPR Detection"):
            run_anpr_detection(
                confidence=confidence, 
                device=device, 
                weights=weights, 
                source="uploaded_video.mp4"
            )

if __name__ == "__main__":
    main()

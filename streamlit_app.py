import streamlit as st
import tempfile
from pathlib import Path
import os
import re

# Set environment before any internal imports happen
os.environ["ENABLE_MULTICLASS"] = "false"

from backend.services.chatbot import run as run_chatbot
from backend.services.llm import OpenRouterGenerator
from backend.core.config import DATABASE_FILE, RAG_INDEX_FILE

@st.cache_resource
def get_generator():
    return OpenRouterGenerator()

st.set_page_config(page_title="Kannada Medicine Assistant", page_icon="💊")

st.title("💊 Kannada Medicine Assistant")
st.markdown("**ಬೆಂಬಲಿತ ಭಾಷೆ (Supported):** ಕನ್ನಡ & English. Upload a medicine cover image and ask a question about it.")

# Predefined Questions
presets = [
    "ಈ ಔಷಧಿಯ ಹೆಸರು ಏನು? (What is the name of this medicine?)",
    "ಈ ಔಷಧಿಯ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ನೀಡಿ. (Give complete information.)",
    "ಈ ಔಷಧಿಯನ್ನು ಯಾವುದಕ್ಕಾಗಿ ಬಳಸುತ್ತಾರೆ? (What is this used for?)",
    "ಇದರ ಮುಖ್ಯ ಉಪಯೋಗ ಏನು? (What is its main use?)",
    "ಈ ಔಷಧಿಯ ಸಂಯೋಜನೆ ಏನು? (What is the composition?)",
    "ಇದರ ಸಾಮಾನ್ಯ ಅಡ್ಡ ಪರಿಣಾಮಗಳು ಯಾವುವು? (What are common side effects?)",
    "ಈ ಔಷಧಿಯ ಡೋಸ್ ಎಷ್ಟು? (What is the dosage?)"
]

uploaded_file = st.file_uploader("Medicine cover image (JPG/PNG)", type=["jpg", "jpeg", "png", "webp", "bmp"])
preset_q = st.selectbox("Select a question (ಅಥವಾ ಕೆಳಗೆ ಟೈಪ್ ಮಾಡಿ)", ["Type my own"] + presets)

if preset_q == "Type my own":
    question = st.text_area("Your question (ನಿಮ್ಮ ಪ್ರಶ್ನೆ)", placeholder="ಇದರ ಉಪಯೋಗ ಏನು?")
else:
    # strip english translation in parens for the actual prompt
    question = preset_q.split(" (")[0]
    st.text_area("Your question (ನಿಮ್ಮ ಪ್ರಶ್ನೆ)", value=question, disabled=True)

if st.button("Identify and answer", type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Processing image and searching catalogue... (\u0c87\u0ca6\u0ca8\u0ccd\u0ca8\u0cc1 \u0c93\u0ca6\u0cb2\u0cbe\u0c97\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6...)"):
            # Save upload to temp file
            with tempfile.TemporaryDirectory() as td:
                temp_dir = Path(td)
                img_path = temp_dir / uploaded_file.name
                img_path.write_bytes(uploaded_file.read())
                
                try:
                    result = run_chatbot(
                        image=img_path,
                        question=question,
                        database=DATABASE_FILE,
                        index=RAG_INDEX_FILE,
                        output=temp_dir / "output",
                        quiet=True,
                        generator=get_generator(),
                        identification_backend="ocr_csv",
                        model=None
                    )
                    
                    # Display Results
                    st.header("Answer")
                    
                    if result["status"] == "identification_uncertain":
                        st.warning("⚠️ " + result["chatbot"]["answer"])
                        st.subheader("Why?")
                        reasons = result.get("identification", {}).get("uncertainty_reasons", [])
                        st.write(", ".join(reasons) if reasons else "Could not confirm identification.")
                    else:
                        st.success(result["chatbot"]["answer"])
                        
                        med = result["identification"].get("final_result", {})
                        if med:
                            with st.expander("Show Medicine Details"):
                                st.write(f"**Medicine:** {med.get('medicine_name')}")
                                st.write(f"**Composition:** {med.get('composition')}")
                                st.write(f"**Manufacturer:** {med.get('manufacturer')}")
                                st.write(f"**Match Score:** {med.get('match_score')}")
                    
                    # Show OCR Regions and Annotated Image
                    with st.expander("Show Technical Details & Read Text"):
                        annotated = result.get("identification", {}).get("stages", {}).get("detection", {}).get("annotated_path")
                        if annotated and Path(annotated).exists():
                            st.image(str(annotated), caption="EasyOCR Bounding Boxes")
                            
                        ocr_regions = result.get("identification", {}).get("stages", {}).get("ocr_csv", {}).get("ocr_regions", [])
                        if ocr_regions:
                            st.write("### Read Text")
                            for r in ocr_regions:
                                st.write(f"- **{r.get('class_name')}**: {r.get('text')}")

                except Exception as e:
                    st.error(f"Error processing request: {e}")

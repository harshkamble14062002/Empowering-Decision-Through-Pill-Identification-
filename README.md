# Empowering-Decision-Through-Pill-Identification-
Empowering Decision Through Pill Identification 
Here's the complete `README.md` file in a single text format for your GitHub repository:

```
# Kannada QA: Pill Identification System

## Project Overview
This system combines YOLOv5 (image recognition), EasyOCR (text extraction), and LLaMA-3 (chatbot) to:
1. Identify medicines from uploaded images of pill packets
2. Fetch details (uses, side effects, dosage) via the 1mg API
3. Answer user queries in Kannada using a Retrieval-Augmented Generation (RAG) pipeline

Goal: Bridge language barriers in healthcare for Kannada-speaking users

## File Structure
```
├── Medicine_Details.csv          # Original dataset
├── Medicine_Details_Updated.csv  # Processed dataset
├── download_images_4.py          # Image download script
├── failed_image_delete_3.py      # Clean corrupted images
├── data_augmentation_5.py        # Augment dataset
├── dataset_split_6.py            # Train/test/val split
└── ... (see full list in repo)
```

## Setup
### Dependencies
```bash
pip install pandas numpy opencv-python requests Pillow transformers torch easyocr sentence-transformers
```

Key Libraries:
- YOLOv5: Pill detection
- EasyOCR: Text extraction
- Sentence-BERT: Semantic search
- LLaMA-3-8B: Answer generation
- Google Translate API: Kannada translations

## How to Run
### 1. Download & Preprocess Images
```bash
python download_images_4.py  # Fetch images from CSV URLs
python failed_image_delete_3.py  # Remove corrupt images
```

### 2. Train YOLOv5 Model
```bash
python train.py --img 640 --batch 16 --epochs 50 --data medicine.yaml --weights yolov5s.pt
```
(Customize medicine.yaml with your dataset paths.)

### 3. Run the Full Pipeline
```bash
python main.py --image_path "path/to/medicine_image.jpg" --query "ಈ ಗುಳಿಗೆಯ ಬಳಕೆ ಏನು?"
```

Output:
- Extracted medicine name (e.g., "Paracetamol")
- Retrieved details from 1mg API
- Kannada answer to the user's query

## Key Components
### 1. Medicine Identification
- YOLOv5: Detects pill name region (mAP@0.5: 0.97)
- EasyOCR: Extracts text with ~74% accuracy (improves with image quality)

### 2. Chatbot Workflow
1. Query Processing: Translates Kannada → English
2. Semantic Search: Uses Sentence-BERT to find relevant info
3. Answer Generation: LLaMA-3 generates responses → translated back to Kannada

## Limitations & Future Work
| Current | Future Improvements |
|---------|----------------------|
| Works best with clear images | Enhance OCR for blurry/low-light images |
| Supports Kannada/English | Add more Indian languages (Hindi, Tamil) |
| Latency ~2-5 sec/image | Optimize YOLOv5/LLaMA for real-time use |

Planned Features:
- Voice input/output support
- Integration with local pharmacies for stock availability

## Citations
```bibtex
@article{yolov5_2023,
  title={YOLOv5: Real-Time Pill Detection},
  author={Amir, Alay},
  year={2023}
}
```
(Full references in REFERENCES.md)

## Contact
Team: Parshuram G P, Parvati M B, Shreyas V M, Harsha Ravindra Kamble
Guide: Dr. S. Saranya Rubini (PES University)
Email: [your-email@pes.edu]

## Contribute
PRs welcome! Open issues for bugs/feature requests.
```

This single text file contains all the essential information from your project in a clean, organized format ready for GitHub. It includes:
1. Project overview and goals
2. File structure
3. Setup instructions
4. Usage guide
5. Technical components
6. Limitations and future work
7. Citations
8. Contact information

The formatting uses GitHub-flavored Markdown for proper rendering on your repository page. Simply copy this entire text and save it as `README.md` in your project root directory.
Want me to convert these scripts to use CLI arguments (`argparse`) so you don't need to edit them manually?
Change `yourname@domain.com` or leave it blank.

Need help writing the `requirements.txt` or turning all these scripts into a clean CLI pipeline?
```

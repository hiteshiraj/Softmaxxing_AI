# ✨ Softmaxxing AI

**An AI-powered facial presentation and profile photo analysis tool.**

Softmaxxing AI uses computer vision and lightweight machine learning to provide actionable, constructive feedback on profile pictures and selfies. It analyzes factors like lighting, posture, image sharpness, and symmetry, giving users practical advice to improve their presentation.

## Features

- **Face Landmark Detection:** Utilizes MediaPipe FaceMesh to map facial structure.
- **Symmetry Analysis:** Heuristic calculations for eye, jaw, and mouth alignment.
- **Posture Checking:** Estimates head tilt and vertical alignment.
- **Lighting Quality:** OpenCV histogram analysis combined with a RandomForest classifier to score lighting quality.
- **Image Sharpness:** Laplacian variance checks to detect blur.
- **Style Clustering:** KMeans clustering to categorize the overall vibe of the photo (e.g., Professional, Casual, Aesthetic).
- **Constructive Feedback Engine:** Generates practical, non-toxic recommendations based on the calculated metrics.

## Tech Stack

- **Frontend/UI:** Streamlit (Custom Dark Mode)
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning / Data Processing:** scikit-learn, NumPy, Pandas
- **Visualizations:** Plotly, OpenCV drawing tools

## Project Structure

```
softmaxxing-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── utils/
│   ├── face_detection.py   # MediaPipe wrappers
│   ├── symmetry.py         # Symmetry heuristics
│   ├── posture.py          # Posture heuristics
│   ├── lighting.py         # Lighting analysis & RF classifier
│   ├── sharpness.py        # Blur detection
│   ├── scoring.py          # Style clustering & overall score
│   ├── recommendations.py  # Feedback logic
│   └── visualization.py    # Overlay drawings
└── dataset/                # Folder for manual dataset collection
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd softmaxxing-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Deployment

This application is designed to be easily deployed on **Streamlit Community Cloud**. 
Simply link your GitHub repository to Streamlit Cloud and set `app.py` as the entrypoint.

## Future Improvements

- Collecting and curating a custom dataset of profile pictures to train a more robust RandomForest classifier for lighting and style.
- Integrating more advanced heuristics for specific aesthetic elements (e.g., color harmony).

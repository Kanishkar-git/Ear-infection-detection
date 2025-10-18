# 👂 AI-Powered Ear Infection Detection System

AI-powered **Ear Infection Detection System** built with **Streamlit**. Upload ear images to detect infections via **Roboflow**, get **LLM-generated medical insights** (symptoms, treatment, prevention), and generate professional **PDF/JSON reports** with patient info, visual results, and clinical analysis.

---

## Features

- **Image Upload & Detection**
  - Upload JPG, JPEG, or PNG ear images.
  - Detects infections and highlights affected areas.
- **AI Medical Insights**
  - Google Gemini LLM generates:
    - Condition overview
    - Symptoms
    - Treatment options
    - Prevention tips
- **Professional Reports**
  - PDF and JSON formats
  - Includes patient details, detection results, clinical insights, and optional doctor notes.
- **User-Friendly Interface**
  - Two-panel layout: Image & Detection / AI Insights
  - Regenerate insights instantly
  - Form to input patient details

---

## Tech Stack

- **Frontend & Web App:** Streamlit  
- **AI Detection:** Roboflow Inference API  
- **LLM Insights:** Google Gemini 2.0  
- **Image Processing:** OpenCV, PIL  
- **Report Generation:** ReportLab  
- **Data Handling:** JSON

---

## Installation

```bash
git clone <repo-url>
cd <project-folder>
pip install -r requirements.txt
streamlit run app.py

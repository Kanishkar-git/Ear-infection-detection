import streamlit as st
from PIL import Image
import tempfile
from inference_sdk import InferenceHTTPClient
from utils import draw_boxes
import google.generativeai as genai
import cv2
import numpy as np
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

# ------------------- Page Config -------------------
st.set_page_config(page_title="Ear Infection Detector", page_icon="👂", layout="wide")

# ------------------- CSS Styling -------------------
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .detection-container {
        display: flex;
        gap: 20px;
    }
    .left-panel {
        flex: 1;
    }
    .right-panel {
        flex: 1;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-point {
        padding: 10px;
        margin: 8px 0;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
    }
    .button-group {
        display: flex;
        gap: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Roboflow Client -------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6zoNpcwVUh76XmUA86ep"
)

# ------------------- Gemini LLM Setup -------------------
genai.configure(api_key="AIzaSyC6NktkmKRpJTMpe10HaLplzAqBXkt7Kgw")

def get_gemini_response(prompt):
    """Fixed version with proper error handling and response access"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
                top_p=0.8,
            )
        )
        
        # Fixed: Access the text properly from response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            return "Unable to generate response - no valid output"
            
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return f"Error generating insight: {str(e)}"

def parse_llm_to_points(text):
    """Convert LLM text response to bullet points"""
    if not text or "Error" in text:
        return ["Information currently unavailable"]
    
    # Remove asterisks and clean up formatting
    text = text.replace('**', '').replace('*', '')
    
    lines = text.split('\n')
    points = []
    
    for line in lines:
        line = line.strip()
        # Remove bullet points, numbers, dashes from the start
        line = line.lstrip('•-–—*123456789.').strip()
        
        # Only keep substantial lines
        if line and len(line) > 10 and not line.startswith('#'):
            points.append(line)
    
    return points[:6]  # Limit to 6 points

def generate_pdf_report(report_data, processed_image=None):
    """Generate a professional PDF medical report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("EAR INFECTION DETECTION REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report Date
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_data['report_date']}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Patient Information Section
    elements.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [
        ['Patient ID:', report_data['patient']['id']],
        ['Name:', report_data['patient']['name']],
        ['Age:', str(report_data['patient']['age'])],
        ['Gender:', report_data['patient']['gender']]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detection Results
    elements.append(Paragraph("DETECTION RESULTS", heading_style))
    for infection in report_data['detection']['infections']:
        elements.append(Paragraph(
            f"• <b>{infection['name']}</b> - Confidence: {infection['confidence']}", 
            styles['Normal']
        ))
    elements.append(Spacer(1, 0.3*inch))
    
    # Clinical Analysis
    elements.append(Paragraph("CLINICAL ANALYSIS", heading_style))
    
    # What is it
    elements.append(Paragraph("<b>Overview:</b>", styles['Normal']))
    elements.append(Paragraph(report_data['analysis'].get('what_is', 'N/A'), styles['Normal']))
    elements.append(Spacer(1, 0.15*inch))
    
    # Symptoms
    elements.append(Paragraph("<b>Symptoms:</b>", styles['Normal']))
    symptoms = parse_llm_to_points(report_data['analysis'].get('symptoms', ''))
    for symptom in symptoms:
        elements.append(Paragraph(f"• {symptom}", styles['Normal']))
    elements.append(Spacer(1, 0.15*inch))
    
    # Treatment
    elements.append(Paragraph("<b>Treatment Options:</b>", styles['Normal']))
    treatments = parse_llm_to_points(report_data['analysis'].get('treatment', ''))
    for treatment in treatments:
        elements.append(Paragraph(f"• {treatment}", styles['Normal']))
    elements.append(Spacer(1, 0.15*inch))
    
    # Prevention
    elements.append(Paragraph("<b>Prevention:</b>", styles['Normal']))
    preventions = parse_llm_to_points(report_data['analysis'].get('prevention', ''))
    for prevention in preventions:
        elements.append(Paragraph(f"• {prevention}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Doctor Notes
    if report_data['doctor_notes'] != "None provided":
        elements.append(Paragraph("DOCTOR'S NOTES", heading_style))
        elements.append(Paragraph(report_data['doctor_notes'], styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    elements.append(Spacer(1, 0.3*inch))
    disclaimer = Paragraph(
        "<i>Disclaimer: This report is generated by an AI system and should be reviewed by a qualified medical professional. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment.</i>",
        styles['Italic']
    )
    elements.append(disclaimer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------- Header -------------------
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.title("👂 Ear Infection Detection System")
st.markdown("*AI-powered detection with instant medical insights*")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Session State -------------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "detected_classes" not in st.session_state:
    st.session_state.detected_classes = []
if "llm_outputs" not in st.session_state:
    st.session_state.llm_outputs = {}
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None

# ------------------- Main Layout -------------------
col_left, col_right = st.columns([1, 1], gap="large")

# ==================== LEFT PANEL ====================
with col_left:
    st.subheader("📸 Image & Detection")
    
    uploaded_file = st.file_uploader("Upload ear image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.uploaded_image = Image.open(uploaded_file)
    
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Run Detection", use_container_width=True):
            with st.spinner("Analyzing image..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    st.session_state.uploaded_image.save(temp_file.name)
                    temp_file_path = temp_file.name

                try:
                    result = client.run_workflow(
                        workspace_name="privacydetailsdetection",
                        workflow_id="custom-workflow-5",
                        images={"image": temp_file_path},
                        use_cache=True
                    )

                    preds = []
                    if isinstance(result, list) and len(result) > 0:
                        for item in result:
                            if isinstance(item, dict) and "predictions" in item:
                                pred_data = item["predictions"]
                                if isinstance(pred_data, dict) and "predictions" in pred_data:
                                    preds.extend(pred_data["predictions"])
                    
                    if preds and len(preds) > 0:
                        try:
                            img_array = np.array(st.session_state.uploaded_image)
                            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            processed_img_bgr = draw_boxes(img_bgr.copy(), preds)
                            processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
                            st.session_state.processed_image = Image.fromarray(processed_img_rgb)
                        except:
                            st.session_state.processed_image = st.session_state.uploaded_image.copy()
                        
                        st.session_state.detected_classes = [p.get('class', 'Unknown') for p in preds]
                        st.session_state.predictions = preds
                        # Reset LLM outputs when new detection is done
                        st.session_state.llm_outputs = {}
                    
                    st.session_state.detection_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")
    
    if st.session_state.detection_done and st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Detection Result", use_column_width=True)

# ==================== RIGHT PANEL ====================
with col_right:
    st.subheader("📊 AI Analysis Results")
    
    if st.session_state.detection_done:
        if st.session_state.detected_classes:
            # Detection Results
            st.markdown("### ✅ Detected Infections")
            for i, p in enumerate(st.session_state.predictions):
                with st.container(border=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{p['class']}**")
                    with col2:
                        st.metric("Confidence", f"{p['confidence']*100:.2f}%")
            
            # Generate LLM Outputs if not already done
            if not st.session_state.llm_outputs:
                with st.spinner("Generating insights..."):
                    # Get class names only (without confidence for cleaner prompts)
                    detected_conditions = ", ".join([p['class'] for p in st.session_state.predictions])
                    
                    # What is it - Overview prompt
                    st.session_state.llm_outputs['what_is'] = get_gemini_response(
                        f"Provide a brief medical explanation of '{detected_conditions}' in 2-3 simple sentences. "
                        f"Focus on what the condition is and why it occurs. Keep it educational and easy to understand."
                    )
                    
                    # Symptoms prompt
                    st.session_state.llm_outputs['symptoms'] = get_gemini_response(
                        f"List exactly 4-5 common symptoms of '{detected_conditions}'. "
                        f"Format: Start each symptom on a new line with a dash (-). "
                        f"Be specific and concise. Example format:\n"
                        f"- Symptom 1 description\n"
                        f"- Symptom 2 description\n"
                        f"Do not include any introduction or conclusion text."
                    )
                    
                    # Treatment prompt
                    st.session_state.llm_outputs['treatment'] = get_gemini_response(
                        f"List exactly 4-5 general care and treatment approaches for '{detected_conditions}'. "
                        f"Format: Start each treatment on a new line with a dash (-). "
                        f"Focus on general care, home remedies, and when to see a doctor. "
                        f"Do NOT recommend specific prescription medications. Example format:\n"
                        f"- Treatment approach 1\n"
                        f"- Treatment approach 2\n"
                        f"Do not include any introduction or conclusion text."
                    )
                    
                    # Prevention prompt
                    st.session_state.llm_outputs['prevention'] = get_gemini_response(
                        f"List exactly 4-5 practical prevention tips for '{detected_conditions}'. "
                        f"Format: Start each tip on a new line with a dash (-). "
                        f"Focus on hygiene, lifestyle, and preventive measures. Example format:\n"
                        f"- Prevention tip 1\n"
                        f"- Prevention tip 2\n"
                        f"Do not include any introduction or conclusion text."
                    )
            
            # Display AI Insights
            if st.session_state.llm_outputs:
                st.markdown("---")
                st.markdown("### 🤖 What is it?")
                st.info(st.session_state.llm_outputs.get('what_is', ''))
                
                st.markdown("### 🔴 Symptoms")
                symptoms = parse_llm_to_points(st.session_state.llm_outputs.get('symptoms', ''))
                if symptoms and symptoms[0] != "Information currently unavailable":
                    for point in symptoms:
                        st.markdown(f"<div class='result-point'>• {point}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Unable to generate symptoms information")
                
                st.markdown("### 💊 General Care & Treatment Options")
                treatments = parse_llm_to_points(st.session_state.llm_outputs.get('treatment', ''))
                if treatments and treatments[0] != "Information currently unavailable":
                    for point in treatments:
                        st.markdown(f"<div class='result-point'>• {point}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Unable to generate treatment information")
                
                st.markdown("### 🛡️ Prevention")
                preventions = parse_llm_to_points(st.session_state.llm_outputs.get('prevention', ''))
                if preventions and preventions[0] != "Information currently unavailable":
                    for point in preventions:
                        st.markdown(f"<div class='result-point'>• {point}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Unable to generate prevention information")
                
                # Add regenerate button
                if st.button("🔄 Regenerate Insights", use_container_width=True):
                    st.session_state.llm_outputs = {}
                    st.rerun()
        else:
            st.warning("No infections detected. Please try another image.")
    else:
        st.info("Upload an image and click 'Run Detection' to analyze")

# ==================== REPORT GENERATION ====================
st.markdown("---")
st.subheader("📋 Generate Medical Report")

if st.session_state.detection_done and st.session_state.detected_classes:
    with st.form("patient_form"):
        st.markdown("### Patient Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID*", placeholder="P-12345")
        with col2:
            patient_name = st.text_input("Patient Name*", placeholder="John Doe")
        with col3:
            patient_age = st.number_input("Age*", min_value=0, max_value=150, value=0)
        
        col4, col5 = st.columns(2)
        with col4:
            patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        with col5:
            doctor_notes = st.text_area("Doctor Notes (Optional)", height=100, placeholder="Additional observations...")
        
        submitted = st.form_submit_button("📥 Generate PDF Report", use_container_width=True, type="primary")
        
        if submitted:
            if not patient_id or not patient_name or patient_age == 0:
                st.error("⚠️ Please fill in all required fields (Patient ID, Name, and Age)")
            else:
                # Generate Report Data
                st.session_state.report_data = {
                    "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient": {
                        "id": patient_id,
                        "name": patient_name,
                        "age": patient_age,
                        "gender": patient_gender
                    },
                    "detection": {
                        "infections": [
                            {
                                "name": p['class'],
                                "confidence": f"{p['confidence']*100:.2f}%"
                            } for p in st.session_state.predictions
                        ]
                    },
                    "analysis": st.session_state.llm_outputs,
                    "doctor_notes": doctor_notes if doctor_notes else "None provided"
                }
                
                # Generate PDF
                try:
                    st.session_state.pdf_buffer = generate_pdf_report(
                        st.session_state.report_data, 
                        st.session_state.processed_image
                    )
                    st.session_state.report_generated = True
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("💡 Make sure reportlab is installed: pip install reportlab")
    
    # Download buttons OUTSIDE the form
    if st.session_state.report_generated and st.session_state.pdf_buffer:
        st.success("✅ Report Generated Successfully!")
        
        col_pdf, col_json = st.columns(2)
        
        with col_pdf:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state.pdf_buffer,
                file_name=f"Medical_Report_{st.session_state.report_data['patient']['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col_json:
            st.download_button(
                label="📄 Download JSON (Backup)",
                data=json.dumps(st.session_state.report_data, indent=2),
                file_name=f"report_{st.session_state.report_data['patient']['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
else:
    st.info("Complete detection first to generate a report")
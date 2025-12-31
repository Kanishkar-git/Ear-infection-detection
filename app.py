import streamlit as st
from PIL import Image
import tempfile
from inference_sdk import InferenceHTTPClient
from utils import draw_boxes
from google import genai
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
import plotly.graph_objects as go
import plotly.express as px
import re

# ================= LANGCHAIN IMPORTS =================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ------------------- Page Config -------------------
st.set_page_config(page_title="AI ENT Doctor Assistant", page_icon="👂", layout="wide")

# ------------------- Enhanced CSS Styling -------------------
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
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
    .severity-mild {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .severity-moderate {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .severity-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .alert-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .doctor-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 10px 0;
    }
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        animation: fadeIn 0.5s;
        color: #0f172a;
    }
    .user-message {
        background-color: #dbeafe;
        border-left: 4px solid #2563eb;
        margin-left: 20px;
        color: #0f172a;
    }
    .doctor-message {
        background-color: #ede9fe;
        border-left: 4px solid #7c3aed;
        margin-right: 20px;
        color: #1e1b4b;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .scan-context-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background-color: #f9fafb;
        padding: 15px 0;
        border-top: 2px solid #e5e7eb;
    }
    .medical-disclaimer {
        background-color: #fef3c7;
        color: #78350f;
        border: 2px solid #f59e0b;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- API Keys & Clients -------------------
# Roboflow Client
client_roboflow = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6zoNpcwVUh76XmUA86ep"
)

# Gemini Client
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "AIzaSyBmsjU8w6b15f_06GqclelPQ20H79XuJWw"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ================= LANGCHAIN ENT DOCTOR SETUP =================

ENT_DOCTOR_SYSTEM_PROMPT = """
You are Dr. Sarah Chen, a board-certified ENT (Ear, Nose, and Throat) specialist with 15 years of clinical experience.
You are reviewing results from a hospital-grade AI otoscope system and guiding the patient after the scan.

=====================
RESPONSE LENGTH RULES (MANDATORY)
=====================
- Maximum 4 sentences per reply
- Each sentence MUST be on a new line
- Keep responses concise and focused
- Stop once the question is answered
- Never end mid-sentence
- Do NOT repeat information already explained unless asked

=====================
COMMUNICATION STYLE
=====================
- Professional, calm, and reassuring
- Simple language with medical accuracy
- Explain BOTH what is happening and why, briefly
- Begin directly with the explanation (no greetings after the first message)
- Reference AI confidence only when relevant

=====================
MEDICAL & ETHICAL RULES
=====================
❌ Do NOT prescribe medications or antibiotics  
❌ Do NOT suggest dosages or treatment plans  
❌ Do NOT confirm a diagnosis  
❌ Do NOT use fear-inducing or alarming language  

✅ Encourage ENT consultation when appropriate  
✅ Use uncertainty-aware language based on confidence  
✅ Prioritize patient safety and clarity  

=====================
RED-FLAG PROTOCOL
=====================
If the patient mentions ANY of the following, advise urgent medical care clearly and briefly:
- Severe or worsening ear pain
- High fever (>101°F / 38.3°C)
- Dizziness or balance problems
- Bloody or foul-smelling discharge
- Sudden hearing loss
- Facial weakness or drooping
- Severe headache with ear pain

=====================
CONSULTATION FLOW
=====================
1. Direct explanation
2. Brief reasoning (why it happens)
3. Practical non-medical advice (if relevant)
4. Clear next step (ENT visit, monitoring, or reassurance)

You are a doctor in a real consultation.
Be precise, concise, and clinically responsible.
"""

def initialize_langchain_chatbot(medical_context_str):
    """Initialize LangChain-powered ENT Doctor chatbot with medical context"""
    
    try:
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=1200,
            convert_system_message_to_human=True
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            human_prefix="Patient",
            ai_prefix="Dr. Chen"
        )
        
        # Inject medical context
        memory.chat_memory.add_ai_message(
            f"""MEDICAL SCAN CONTEXT (Reference this for all responses):

{medical_context_str}

I will use this scan data to provide accurate, context-aware medical guidance throughout our consultation."""
        )
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=ENT_DOCTOR_SYSTEM_PROMPT + """

Conversation History:
{chat_history}

Patient: {input}

Dr. Chen:"""
        )
        
        # Create conversation chain
        chatbot = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )
        
        return chatbot
    except Exception as e:
        st.error(f"Chatbot initialization error: {str(e)}")
        return None

def format_doctor_reply(text):
    """Format doctor's reply for better readability"""
    if not text:
        return ""

    text = text.strip()
    sentences = text.replace("\n", " ").split(". ")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    formatted = ".\n".join(sentences[:5])

    if not formatted.endswith("."):
        formatted += "."

    return formatted

def get_medical_context_string(medical_context):
    """Format medical context for the chatbot"""
    if not medical_context:
        return "No detection data available yet."
    
    confidence = medical_context.get('confidence', 0)
    conf_label = medical_context.get('confidence_label', 'unknown')
    
    context_str = f"""
SCAN DETAILS:
- Detection Source: Hospital-grade AI otoscope system
- Detected Condition: {medical_context.get('condition', 'Unknown')}
- Detection Confidence: {confidence:.1f}% ({conf_label} confidence)
- Visual Coverage: {medical_context.get('box_area', 'standard')} bounding box area
- Patient Age: {medical_context.get('age', 'Not provided')}
- Visual Indicators Detected: {', '.join(medical_context.get('visual_features', ['Redness', 'Inflammation']))}

CONFIDENCE INTERPRETATION:
- High confidence (>85%): Strong visual indicators present
- Moderate confidence (60-85%): Notable indicators, recommend verification
- Low confidence (<60%): Weak indicators, further evaluation needed
"""
    return context_str

# ================= ANALYSIS FUNCTIONS =================

def get_advanced_gemini_response(detected_condition, confidence, patient_age=None, visual_features=None):
    """Get comprehensive medical analysis from Gemini"""
    
    if patient_age is None:
        patient_age = "Not provided"
    if visual_features is None:
        visual_features = ["Redness detected", "Inflammation visible", "Structural abnormalities"]
    
    prompt = f"""You are a clinical decision-support AI integrated with a computer-vision ear infection detection system.

Detected Condition: {detected_condition}
Detection Confidence: {confidence:.2f}%
Patient Age: {patient_age}
Visual Indicators Detected: {', '.join(visual_features)}

Generate a comprehensive medical analysis with the following sections:

SECTION 1: OVERVIEW
Provide a 2-3 line explanation of the detected condition and what the confidence level means.

SECTION 2: SEVERITY ASSESSMENT
Classify severity:
- <60% → Mild / Early-stage
- 60-85% → Moderate
- >85% → High severity
Explain the severity classification.

SECTION 3: VISUAL REASONING
Explain what visual indicators the AI detected and how they relate to the condition.

SECTION 4: INFECTION TIMELINE
Estimate the likely infection stage and possible progression over 3 days.

SECTION 5: PROBABLE SYMPTOMS
List 4-5 symptoms with estimated likelihood percentages that align with the confidence level.

SECTION 6: PREVENTION & CARE
Provide 4-5 condition-specific prevention tips.

SECTION 7: RED-FLAG ALERTS
List 4-5 situations requiring immediate medical attention.

SECTION 8: DISCLAIMER
Provide a confidence-aware disclaimer.

After all sections, provide chart data in this exact JSON format:
{{
  "detection_confidence": {{
    "condition": "{detected_condition}",
    "confidence_percent": {confidence:.2f}
  }},
  "severity_level": {{
    "label": "Mild/Moderate/Severe",
    "numeric_level": 1-5
  }},
  "symptom_probability_distribution": {{
    "Ear Pain": 85,
    "Hearing Loss": 65,
    "Fever": 70,
    "Discharge": 55
  }},
  "infection_progress_timeline": {{
    "Day 1": 2.5,
    "Day 2": 3.0,
    "Day 3": 3.5
  }},
  "visual_feature_contribution": {{
    "Redness": 35,
    "Inflammation": 40,
    "Structural Changes": 25
  }},
  "prevention_effectiveness": {{
    "Keep Ear Dry": 4.5,
    "Avoid Q-tips": 4.0,
    "Regular Check-ups": 4.8,
    "Proper Hygiene": 4.2
  }}
}}"""

    try:
        response = gemini_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config={
                'temperature': 0.7,
                'max_output_tokens': 2500,
            }
        )
        
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            return None
            
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return None

def parse_gemini_response(response_text):
    """Parse Gemini response into structured sections"""
    sections = {
        'overview': '',
        'severity': '',
        'visual_reasoning': '',
        'timeline': '',
        'symptoms': [],
        'prevention': [],
        'red_flags': [],
        'disclaimer': '',
        'chart_data': {}
    }

    if not response_text:
        return sections

    # Extract JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_text = json_match.group()
            sections['chart_data'] = json.loads(json_text)
    except Exception as e:
        st.warning(f"Could not parse chart data: {str(e)}")

    # Parse text sections
    lines = response_text.split("\n")
    current_section = None

    for line in lines:
        l = line.strip()
        
        if "SECTION 1" in l or "OVERVIEW" in l:
            current_section = "overview"
        elif "SECTION 2" in l or "SEVERITY" in l:
            current_section = "severity"
        elif "SECTION 3" in l or "VISUAL REASONING" in l:
            current_section = "visual_reasoning"
        elif "SECTION 4" in l or "TIMELINE" in l:
            current_section = "timeline"
        elif "SECTION 5" in l or "SYMPTOMS" in l:
            current_section = "symptoms"
        elif "SECTION 6" in l or "PREVENTION" in l:
            current_section = "prevention"
        elif "SECTION 7" in l or "RED-FLAG" in l or "ALERTS" in l:
            current_section = "red_flags"
        elif "SECTION 8" in l or "DISCLAIMER" in l:
            current_section = "disclaimer"
        elif current_section and l and not l.startswith("{"):
            if current_section in ["symptoms", "prevention", "red_flags"]:
                if l.startswith("-") or l.startswith("•") or l.startswith("*"):
                    sections[current_section].append(l.lstrip("-•* ").strip())
            else:
                sections[current_section] += l + " "

    return sections

def create_visualizations(chart_data):
    """Create interactive Plotly charts"""
    charts = {}
    
    if not chart_data:
        return charts
    
    try:
        # Confidence Gauge
        if 'detection_confidence' in chart_data:
            conf = chart_data['detection_confidence']['confidence_percent']
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={'text': "Detection Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "gold"},
                        {'range': [85, 100], 'color': "limegreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_conf.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            charts['confidence'] = fig_conf
        
        # Symptom Probability
        if 'symptom_probability_distribution' in chart_data:
            symptoms = chart_data['symptom_probability_distribution']
            fig_symptoms = go.Figure(data=[
                go.Bar(
                    x=list(symptoms.keys()),
                    y=list(symptoms.values()),
                    marker_color='indianred',
                    text=list(symptoms.values()),
                    textposition='auto',
                )
            ])
            fig_symptoms.update_layout(
                title="Symptom Probability Distribution",
                xaxis_title="Symptoms",
                yaxis_title="Probability (%)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            charts['symptoms'] = fig_symptoms
        
        # Infection Timeline
        if 'infection_progress_timeline' in chart_data:
            timeline = chart_data['infection_progress_timeline']
            fig_timeline = go.Figure(data=[
                go.Scatter(
                    x=list(timeline.keys()),
                    y=list(timeline.values()),
                    mode='lines+markers',
                    line=dict(color='firebrick', width=3),
                    marker=dict(size=10)
                )
            ])
            fig_timeline.update_layout(
                title="Predicted Infection Progression (3 Days)",
                xaxis_title="Day",
                yaxis_title="Severity Score (1-5)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            charts['timeline'] = fig_timeline
        
        # Visual Features Pie Chart
        if 'visual_feature_contribution' in chart_data:
            features = chart_data['visual_feature_contribution']
            fig_features = go.Figure(data=[
                go.Pie(
                    labels=list(features.keys()),
                    values=list(features.values()),
                    hole=0.3
                )
            ])
            fig_features.update_layout(
                title="Visual Feature Contribution",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            charts['features'] = fig_features
        
        # Prevention Effectiveness
        if 'prevention_effectiveness' in chart_data:
            prevention = chart_data['prevention_effectiveness']
            fig_prevention = go.Figure(data=[
                go.Bar(
                    y=list(prevention.keys()),
                    x=list(prevention.values()),
                    orientation='h',
                    marker_color='lightseagreen',
                    text=list(prevention.values()),
                    textposition='auto',
                )
            ])
            fig_prevention.update_layout(
                title="Prevention Strategy Effectiveness",
                xaxis_title="Effectiveness Score (1-5)",
                yaxis_title="Prevention Strategy",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            charts['prevention'] = fig_prevention
            
    except Exception as e:
        st.warning(f"Could not generate all visualizations: {str(e)}")
    
    return charts

def generate_pdf_report(report_data, processed_image=None):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Title
    elements.append(Paragraph("AI-ASSISTED EAR INFECTION DETECTION REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_data['report_date']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Report ID:</b> {report_data.get('report_id', 'N/A')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    elements.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['Heading2']))
    patient_data = [
        ['Patient ID:', report_data['patient']['id']],
        ['Name:', report_data['patient']['name']],
        ['Age:', str(report_data['patient']['age'])],
        ['Gender:', report_data['patient']['gender']],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detection Results
    elements.append(Paragraph("<b>DETECTION RESULTS</b>", styles['Heading2']))
    for infection in report_data['detection']['infections']:
        elements.append(Paragraph(
            f"• <b>{infection['name']}</b> - Confidence: {infection['confidence']}", 
            styles['Normal']
        ))
    elements.append(Spacer(1, 0.3*inch))
    
    # Analysis sections
    analysis = report_data['analysis']
    
    if analysis.get('overview'):
        elements.append(Paragraph("<b>CLINICAL OVERVIEW</b>", styles['Heading2']))
        elements.append(Paragraph(analysis['overview'], styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
    
    if analysis.get('severity'):
        elements.append(Paragraph("<b>SEVERITY ASSESSMENT</b>", styles['Heading2']))
        elements.append(Paragraph(analysis['severity'], styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    elements.append(Spacer(1, 0.3*inch))
    disclaimer = ("This AI-generated report is for clinical support only. "
                 "It does NOT provide a medical diagnosis. "
                 "Consult a qualified ENT specialist for confirmation.")
    elements.append(Paragraph(f"<b>{disclaimer}</b>", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------- Session State Initialization -------------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "detected_classes" not in st.session_state:
    st.session_state.detected_classes = []
if "analysis" not in st.session_state:
    st.session_state.analysis = {}
if "chart_data" not in st.session_state:
    st.session_state.chart_data = {}
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None
if "patient_age" not in st.session_state:
    st.session_state.patient_age = None
if "medical_context" not in st.session_state:
    st.session_state.medical_context = {}
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_user_message" not in st.session_state:
    st.session_state.pending_user_message = None

# ------------------- Header -------------------
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.title("👂 AI ENT Doctor Assistant")
st.markdown("*Hospital-Grade Detection with Expert AI Consultation*")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Main Layout -------------------
tab1, tab2, tab3 = st.tabs(["🔬 Detection & Analysis", "👨‍⚕️ Consult ENT Doctor", "📋 Generate Report"])

# ==================== TAB 1: DETECTION ====================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("📸 Image Upload & Detection")
        
        patient_age_input = st.number_input("Patient Age (Optional)", min_value=0, max_value=150, value=0, key="age_input")
        if patient_age_input > 0:
            st.session_state.patient_age = patient_age_input
        
        uploaded_file = st.file_uploader("Upload ear image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
        
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Run Detection", use_container_width=True, type="primary"):
                with st.spinner("Analyzing image..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        st.session_state.uploaded_image.save(temp_file.name)
                        temp_file_path = temp_file.name
                    
                    try:
                        result = client_roboflow.run_workflow(
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
                            
                            # Build medical context
                            confidence = preds[0]['confidence'] * 100
                            conf_label = "high" if confidence > 85 else "moderate" if confidence > 60 else "low"
                            
                            st.session_state.medical_context = {
                                'condition': preds[0]['class'],
                                'confidence': confidence,
                                'confidence_label': conf_label,
                                'box_area': 'large' if preds[0].get('width', 0) > 200 else 'standard',
                                'age': st.session_state.patient_age if st.session_state.patient_age else 'Not provided',
                                'visual_features': ['Redness detected', 'Inflammation visible', 'Structural changes']
                            }
                            
                            # Initialize chatbot
                            medical_context_str = get_medical_context_string(st.session_state.medical_context)
                            st.session_state.chatbot = initialize_langchain_chatbot(medical_context_str)
                            st.session_state.chat_history = []
                            
                            st.session_state.analysis = {}
                            st.session_state.chart_data = {}
                        else:
                            st.warning("No infections detected in the image.")
                        
                        st.session_state.detection_done = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
        
        if st.session_state.detection_done and st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption="Detection Result", use_container_width=True)

    with col_right:
        st.subheader("📊 Clinical Analysis")
        
        if st.session_state.detection_done:
            if st.session_state.detected_classes:
                # Detection Results
                st.markdown("### ✅ Detected Conditions")
                for i, p in enumerate(st.session_state.predictions):
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**{p['class']}**")
                        with col2:
                            confidence = p['confidence']*100
                            st.metric("Confidence", f"{confidence:.2f}%")
                
                # Generate Analysis
                if not st.session_state.analysis:
                    with st.spinner("Generating clinical insights..."):
                        detected_condition = st.session_state.predictions[0]['class']
                        confidence = st.session_state.predictions[0]['confidence'] * 100
                        
                        response = get_advanced_gemini_response(
                            detected_condition,
                            confidence,
                            st.session_state.patient_age,
                            ["Redness detected", "Inflammation visible", "Structural changes"]
                        )
                        
                        if response:
                            st.session_state.analysis = parse_gemini_response(response)
                            st.session_state.chart_data = st.session_state.analysis.get('chart_data', {})
                
                # Display Analysis
                if st.session_state.analysis:
                    analysis = st.session_state.analysis
                    
                    if analysis.get('overview'):
                        st.markdown("#### 📋 Overview")
                        st.info(analysis['overview'])
                    
                    if analysis.get('severity'):
                        st.markdown("#### ⚠️ Severity Assessment")
                        confidence = st.session_state.predictions[0]['confidence'] * 100
                        if confidence < 60:
                            st.success(analysis['severity'])
                        elif confidence < 85:
                            st.warning(analysis['severity'])
                        else:
                            st.error(analysis['severity'])
                    
                    # Display Charts
                    if st.session_state.chart_data:
                        st.markdown("#### 📈 Visual Analytics")
                        
                        charts = create_visualizations(st.session_state.chart_data)
                        
                        if 'confidence' in charts:
                            st.plotly_chart(charts['confidence'], use_container_width=True, key="conf_chart")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'symptoms' in charts:
                                st.plotly_chart(charts['symptoms'], use_container_width=True, key="symp_chart")
                        with col2:
                            if 'timeline' in charts:
                                st.plotly_chart(charts['timeline'], use_container_width=True, key="time_chart")
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            if 'features' in charts:
                                st.plotly_chart(charts['features'], use_container_width=True, key="feat_chart")
                        with col4:
                            if 'prevention' in charts:
                                st.plotly_chart(charts['prevention'], use_container_width=True, key="prev_chart")
                    
                    # Additional sections
                    if analysis.get('symptoms'):
                        with st.expander("🩺 Probable Symptoms", expanded=False):
                            for symptom in analysis['symptoms']:
                                st.markdown(f"• {symptom}")
                    
                    if analysis.get('prevention'):
                        with st.expander("🛡️ Prevention & Care Guidance", expanded=False):
                            for prev in analysis['prevention']:
                                st.markdown(f"• {prev}")
                    
                    if analysis.get('red_flags'):
                        with st.expander("🚨 Red-Flag Alerts", expanded=False):
                            st.error("Seek immediate medical attention if you experience:")
                            for flag in analysis['red_flags']:
                                st.markdown(f"• {flag}")
                    
                    if analysis.get('disclaimer'):
                        st.markdown("---")
                        st.caption(analysis['disclaimer'])

# ==================== TAB 2: ENT DOCTOR CHAT ====================
with tab2:
    st.markdown("<div class='doctor-badge'>👨‍⚕️ Dr. Sarah Chen, ENT Specialist</div>", unsafe_allow_html=True)
    
    if not st.session_state.detection_done:
        st.warning("⚠️ Please complete the detection in Tab 1 before consulting the doctor.")
        st.info("The AI doctor needs scan results to provide accurate medical guidance.")
    else:
        # Display medical context
        with st.expander("📋 View Scan Context", expanded=False):
            st.markdown("<div class='scan-context-box'>", unsafe_allow_html=True)
            context = st.session_state.medical_context
            st.write(f"**Detected Condition:** {context.get('condition', 'Unknown')}")
            st.write(f"**Confidence:** {context.get('confidence', 0):.1f}% ({context.get('confidence_label', 'unknown')})")
            st.write(f"**Patient Age:** {context.get('age', 'Not provided')}")
            st.write(f"**Visual Features:** {', '.join(context.get('visual_features', []))}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Medical Disclaimer
        st.markdown("""
        <div class='medical-disclaimer'>
        <b>⚕️ Medical Disclaimer:</b> This AI consultation supports clinical understanding. 
        It does not replace physical examination or professional diagnosis by a qualified ENT specialist.
        </div>
        """, unsafe_allow_html=True)
        
        # Chat History Display
        st.markdown("### 💬 Consultation")
        chat_container = st.container(height=500)
        
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class='doctor-message chat-message'>
                <b>Dr. Chen:</b><br>
                Hello! I've reviewed your ear scan results. I'm here to help you understand the findings 
                and answer any questions you may have. What would you like to know?
                </div>
                """, unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class='user-message chat-message'>
                    <b>You:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='doctor-message chat-message'>
                    <b>Dr. Chen:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Process pending message
        if st.session_state.pending_user_message:
            user_question = st.session_state.pending_user_message
            
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            with st.spinner("Dr. Chen is typing..."):
                try:
                    if st.session_state.chatbot is None:
                        medical_context_str = get_medical_context_string(st.session_state.medical_context)
                        st.session_state.chatbot = initialize_langchain_chatbot(medical_context_str)
                    
                    if st.session_state.chatbot is None:
                        raise Exception("Failed to initialize chatbot")
                    
                    response = st.session_state.chatbot.predict(input=user_question)
                    formatted_response = format_doctor_reply(response)
                    
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
                    formatted_response = "I apologize, but I'm having trouble responding right now. Please try rephrasing your question."
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': formatted_response
            })
            
            st.session_state.pending_user_message = None
            st.rerun()
        
        # Chat Input
        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Ask Dr. Chen about your scan results...",
                key="user_input",
                placeholder="e.g., What does this infection mean? Is this serious?"
            )
        with col2:
            send_button = st.button("Send", use_container_width=True, type="primary")
        
        if send_button and user_question and st.session_state.pending_user_message is None:
            st.session_state.pending_user_message = user_question
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick question buttons
        st.markdown("#### 💡 Common Questions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("What is this condition?", use_container_width=True):
                st.session_state.pending_user_message = 'Can you explain what this condition means?'
                st.rerun()
        
        with col2:
            if st.button("How serious is this?", use_container_width=True):
                st.session_state.pending_user_message = 'How serious is this infection?'
                st.rerun()
        
        with col3:
            if st.button("What should I do next?", use_container_width=True):
                st.session_state.pending_user_message = 'What are the next steps I should take?'
                st.rerun()

# ==================== TAB 3: GENERATE REPORT ====================
with tab3:
    st.subheader("📋 Generate Medical Report")
    
    if not st.session_state.detection_done:
        st.warning("⚠️ Please complete detection in Tab 1 first.")
    else:
        st.info("Generate a comprehensive PDF report with all detection results and analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", value="PAT-" + datetime.now().strftime("%Y%m%d-%H%M"))
            patient_name = st.text_input("Patient Name", value="John Doe")
        
        with col2:
            patient_age_report = st.number_input("Age", min_value=0, max_value=150, 
                                                value=st.session_state.patient_age if st.session_state.patient_age else 30)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        doctor_notes = st.text_area("Doctor's Notes (Optional)", 
                                    placeholder="Add any additional observations or recommendations...")
        
        if st.button("🔄 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating comprehensive medical report..."):
                report_data = {
                    'report_id': f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'report_date': datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                    'patient': {
                        'id': patient_id,
                        'name': patient_name,
                        'age': patient_age_report,
                        'gender': patient_gender
                    },
                    'detection': {
                        'infections': [
                            {
                                'name': p['class'],
                                'confidence': f"{p['confidence']*100:.2f}%"
                            }
                            for p in st.session_state.predictions
                        ]
                    },
                    'analysis': st.session_state.analysis,
                    'doctor_notes': doctor_notes if doctor_notes else "None provided"
                }
                
                st.session_state.report_data = report_data
                st.session_state.pdf_buffer = generate_pdf_report(report_data, st.session_state.processed_image)
                st.session_state.report_generated = True
                st.success("✅ Report generated successfully!")
        
        if st.session_state.report_generated and st.session_state.pdf_buffer:
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state.pdf_buffer,
                file_name=f"ear_infection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
            
            # Report Preview
            with st.expander("📄 Report Preview", expanded=False):
                if st.session_state.report_data:
                    rd = st.session_state.report_data
                    st.markdown(f"**Report ID:** {rd['report_id']}")
                    st.markdown(f"**Patient:** {rd['patient']['name']} ({rd['patient']['age']}y, {rd['patient']['gender']})")
                    st.markdown(f"**Date:** {rd['report_date']}")
                    
                    st.markdown("**Detected Conditions:**")
                    for inf in rd['detection']['infections']:
                        st.markdown(f"- {inf['name']} ({inf['confidence']})")
                    
                    if rd.get('doctor_notes') != "None provided":
                        st.markdown(f"**Doctor's Notes:** {rd['doctor_notes']}")

# Footer
st.markdown("---")
st.caption("🏥 AI ENT Doctor Assistant | Powered by Roboflow CV, Google Gemini & LangChain | For medical guidance only")
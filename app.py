import os
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, session, send_from_directory
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import openai
import json
from openai import OpenAI
import math
import time
import uuid
import threading
from queue import Queue
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import PyPDF2
import easyocr
import numpy as np
import PIL.Image as Image
import io
import re
import asyncio
import concurrent.futures
from functools import partial
import aiohttp
import cv2
import jinja2  # Add jinja2 import to handle Undefined objects
from collections import Counter
import base64

def is_valid_uuid(val):
    """Check if a string is a valid UUID"""
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Session timeout after 30 minutes of inactivity
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes in seconds
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add middleware to handle large requests
@app.before_request
def limit_request_size():
    """Prevent requests that are too large"""
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'Request too large'}), 413
    
    # Check URL length to prevent 414 errors
    if len(request.url) > 8192:  # Most servers limit URLs to 8KB
        return jsonify({'error': 'URL too long. Please use POST request instead.'}), 414

# Simple User model
class User(UserMixin):
    def __init__(self, id, email, password_hash):
        self.id = id
        self.email = email
        self.password_hash = password_hash

# Mock database - in production, use a real database
users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# configure your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# ------------------------------------------------------------------
# DeepSeek v3 client (using OpenAI-compatible SDK)
DEESEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "sk-41f7a721ad3b4d4faf36fc818ad3597f")
DEESEEK_API_BASE  = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
deepseek_client   = OpenAI(
    api_key=DEESEEK_API_KEY,
    base_url=DEESEEK_API_BASE
)

# OpenAI client for Vision API (DeepSeek doesn't have vision capabilities)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Configure which OpenAI vision model to use (GPT-4.1 Mini is the latest and most cost-effective)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")  # Options: gpt-4.1-mini, gpt-4o-mini, gpt-4o, gpt-4-vision-preview
# ------------------------------------------------------------------

# Add this to enable offline mode - set to True when on restricted networks
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "False").lower() in ("true", "1", "t")

# Mock data for offline mode
MOCK_SUMMARY = """This contract outlines an agreement between Company A and Client B for software development services. 
Key terms include a project timeline of 6 months, payment schedule of $10,000 per month, and intellectual property rights 
belonging to Client B upon final payment. There are provisions for termination with 30 days notice, confidentiality 
requirements extending 2 years beyond contract end, and limited liability capped at total contract value."""

MOCK_CLAUSES = [
    {
        "id": 1,
        "type": "Payment Default Penalties",
        "risk": "High",
        "risk_score": 78,
        "risk_category": "Unsafe",
        "clause": "Late payment triggers compound interest and immediate acceleration of all amounts due",
        "consequences": "Could result in immediate demand for full contract value plus penalties",
        "amount": "$10,000 monthly + 5% compound interest",
        "deadline": "5 business days grace period",
        "responsible_party": "Client B",
        "likelihood": "Moderate - depends on cash flow management",
        "mitigation": "Negotiate longer grace period or cap on penalties",
        "scoring_breakdown": {
            "financial_impact": 85,
            "business_disruption": 70,
            "legal_risk": 75,
            "likelihood": 60,
            "mitigation_difficulty": 80,
            "calculation": "85×0.3 + 70×0.25 + 75×0.2 + 60×0.15 + 80×0.1 = 78"
        }
    },
    {
        "id": 2,
        "type": "Termination & Exit Costs",
        "risk": "Medium",
        "risk_score": 52,
        "risk_category": "Warning",
        "clause": "Either party may terminate with 30 days notice, but client must pay for completed work and cover transition costs",
        "consequences": "Project could end prematurely with additional transition expenses",
        "amount": "Payment for work completed + transition costs",
        "deadline": "30 days notice required",
        "responsible_party": "Both parties",
        "likelihood": "Low - typically occurs only in dispute situations",
        "mitigation": "Clarify transition cost definitions and caps",
        "scoring_breakdown": {
            "financial_impact": 45,
            "business_disruption": 60,
            "legal_risk": 40,
            "likelihood": 35,
            "mitigation_difficulty": 50,
            "calculation": "45×0.3 + 60×0.25 + 40×0.2 + 35×0.15 + 50×0.1 = 52"
        }
    },
    {
        "id": 3,
        "type": "Intellectual Property Transfer Risk",
        "risk": "Low",
        "risk_score": 25,
        "risk_category": "Safe",
        "clause": "All intellectual property transfers to Client B upon final payment completion",
        "consequences": "No IP rights retained until full payment received",
        "amount": "Full IP portfolio value",
        "deadline": "Upon final payment",
        "responsible_party": "Company A",
        "likelihood": "Very low - standard payment completion risk",
        "mitigation": "Establish interim IP licensing during development",
        "scoring_breakdown": {
            "financial_impact": 30,
            "business_disruption": 20,
            "legal_risk": 25,
            "likelihood": 15,
            "mitigation_difficulty": 35,
            "calculation": "30×0.3 + 20×0.25 + 25×0.2 + 15×0.15 + 35×0.1 = 25"
        }
    }
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variable to track analysis freshness
current_analysis_time = 0

# Add these global variables at the top
status_queue = Queue()
status_clients = []
processing_status = {}

# Add this near the top with your other global variables
document_processing_status = {}

# After document_processing_status declaration (around line 64)

def send_status_update(status_id, status, message, elapsed=0):
    """Helper to send status updates during processing"""
    status_data = {
        'id': status_id,
        'status': status,
        'message': message,
        'elapsed': elapsed,
        'timestamp': time.time()
    }
    status_queue.put(status_data)
    return status_data

# User routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('landing.html')

@app.route('/upload')
@login_required
def upload():
    """Upload page route"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload"""
    try:
        # Check if the correct file field exists
        if 'contract' not in request.files:
            flash('No file selected')
            return redirect(url_for('upload'))
        
        file = request.files['contract']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('upload'))
        
        if file and file.filename.lower().endswith('.pdf'):
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            unique_filename = f"{timestamp}_{filename}"
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Initialize processing status
            document_processing_status[document_id] = {
                'status': 'uploaded',
                'progress': 0,
                'current_step': 'File uploaded successfully',
                'complete': False,
                'filename': filename,
                'filepath': filepath,
                'start_time': time.time(),
                'user_id': current_user.id
            }
            
            # Start processing in background
            processing_thread = threading.Thread(
                target=process_document,
                args=(filepath, document_id)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
            # Redirect to analyzing page
            return redirect(url_for('analyzing', document_id=document_id))
        else:
            flash('Please upload a PDF file')
            return redirect(url_for('upload'))
            
    except Exception as e:
        flash(f'Error uploading file: {str(e)}')
        print(f"Upload error: {e}")  # For debugging
        return redirect(url_for('upload'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists
        for user_id, user in users.items():
            if user.email == email:
                if check_password_hash(user.password_hash, password):
                    login_user(user)
                    next_page = request.args.get('next')
                    return redirect(next_page or url_for('dashboard'))
                flash('Invalid password')
                return redirect(url_for('login'))
        
        # If we get here, user doesn't exist
        flash('User not found')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        for user in users.values():
            if user.email == email:
                flash('User already exists')
                return redirect(url_for('register'))
        
        # Create new user
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        users[user_id] = User(user_id, email, password_hash)
        
        # Log in the new user
        login_user(users[user_id])
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # List all user's documents
    user_documents = []
    for doc_id, status in document_processing_status.items():
        # Filter by current user's documents and only include completed analyses
        if status.get('user_id') == current_user.id and status.get('complete', False):
            user_documents.append({
                'id': doc_id,
                'filename': status.get('filename', ''),
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(status.get('start_time', 0))),
                'high_risk': len([c for c in status.get('clauses', []) if c.get('risk') == 'High']),
                'medium_risk': len([c for c in status.get('clauses', []) if c.get('risk') == 'Medium']),
                'low_risk': len([c for c in status.get('clauses', []) if c.get('risk') == 'Low'])
            })
    
    return render_template('dashboard.html', documents=user_documents)

def extract_text_with_easyocr(filepath, languages=['en']):
    """
    Extract text from a PDF or image using EasyOCR with enhanced preprocessing.
    More powerful for scanned documents with non-uniform formatting.
    
    Args:
        filepath: Path to the PDF file
        languages: List of language codes to detect (default: ['en'] for English)
                  Other examples: ['en', 'ch_sim'] for English and Simplified Chinese
                                 ['en', 'fr'] for English and French
    
    Returns:
        Extracted text as a string
    """
    try:
        # Initialize the OCR reader with optimized settings
        reader = easyocr.Reader(languages, gpu=False)  # Use CPU for stability
        
        # Convert PDF to images with enhanced settings
        images = convert_from_path(
            filepath, 
            dpi=300,  # Higher DPI for better text recognition
            fmt='JPEG',
            thread_count=2
        )
        
        full_text = []
        
        # Process each page with enhanced image preprocessing
        for page_num, img in enumerate(images):
            try:
                # Convert PIL image to numpy array
                img_np = np.array(img)
                
                # Enhanced image preprocessing for better OCR
                # Convert to grayscale if needed
                if len(img_np.shape) == 3:
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_np
                
                # Apply image enhancement techniques
                # 1. Noise reduction
                img_denoised = cv2.fastNlMeansDenoising(img_gray)
                
                # 2. Contrast enhancement using CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_enhanced = clahe.apply(img_denoised)
                
                # 3. Morphological operations to clean up text
                kernel = np.ones((1,1), np.uint8)
                img_processed = cv2.morphologyEx(img_enhanced, cv2.MORPH_CLOSE, kernel)
                
                # Run OCR on the processed image with optimized parameters
                results = reader.readtext(
                    img_processed,
                    detail=1,  # Get bounding box information
                    paragraph=True,  # Group text into paragraphs
                    width_ths=0.7,  # Text width threshold
                    height_ths=0.7,  # Text height threshold
                    mag_ratio=1.5,  # Magnification ratio
                    slope_ths=0.1,  # Slope threshold for text detection
                    ycenter_ths=0.5,  # Y-center threshold
                    low_text=0.4,  # Low text threshold
                    link_threshold=0.4,  # Link threshold
                    canvas_size=2560,  # Canvas size
                    text_threshold=0.7  # Text confidence threshold
                )
                
                # Extract text with confidence filtering
                page_text = []
                for detection in results:
                    if len(detection) >= 3:
                        text = detection[1]
                        confidence = detection[2]
                        
                        # Only include text with reasonable confidence
                        if confidence > 0.5 and len(text.strip()) > 1:
                            # Clean up the detected text
                            cleaned_text = clean_ocr_text(text)
                            if cleaned_text:
                                page_text.append(cleaned_text)
                
                # Join all text blocks from this page
                if page_text:
                    page_content = " ".join(page_text)
                    full_text.append(page_content)
                    print(f"Page {page_num + 1}: Extracted {len(page_text)} text blocks")
                else:
                    print(f"Page {page_num + 1}: No reliable text detected")
                    
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                continue
        
        # Join all pages with double newlines
        result_text = "\n\n".join(full_text)
        
        # Post-process the entire text for better formatting
        result_text = post_process_ocr_text(result_text)
        
        print(f"OCR completed: {len(result_text)} characters extracted from {len(images)} pages")
        return result_text
        
    except Exception as e:
        print(f"Error in EasyOCR extraction: {e}")
        return ""

def clean_ocr_text(text):
    """Clean up individual OCR text detections."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove very short or likely garbage text
    if len(text) < 2:
        return ""
    
    # Remove text that's mostly special characters
    if len(re.sub(r'[^\w\s]', '', text)) < len(text) * 0.3:
        return ""
    
    # Fix common OCR errors
    text = text.replace('|', 'I')  # Common OCR error
    text = text.replace('0', 'O')  # In text context
    text = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1\2', text)  # Fix split numbers/letters
    
    return text

def post_process_ocr_text(text):
    """Post-process the entire OCR text for better formatting."""
    # Fix common OCR formatting issues
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix sentence spacing
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces between words
    
    # Fix monetary amounts that might have been split by OCR
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    text = re.sub(r'(\d)\s+,\s+(\d)', r'\1,\2', text)
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
    
    # Fix percentage signs
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    
    # Fix common legal/contract terms that might be split by OCR
    contract_terms = [
        (r'term\s+ination', 'termination'),
        (r'liab\s+ility', 'liability'),
        (r'indemn\s+ity', 'indemnity'),
        (r'copy\s+right', 'copyright'),
        (r'trade\s+mark', 'trademark'),
        (r'intel\s+lectual', 'intellectual'),
        (r'confiden\s+tial', 'confidential'),
        (r'RENT\s+AL\s+AGREE\s+MENT', 'RENTAL AGREEMENT'),
        (r'AGREE\s+MENT', 'AGREEMENT'),
        (r'TERMIN\s+ATION', 'TERMINATION'),
        (r'Month\s+ly', 'Monthly'),
        (r'depos\s+it', 'deposit')
    ]
    
    for pattern, replacement in contract_terms:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Fix spaced out monetary amounts more aggressively
    # Pattern: $ 2 , 5 0 0 . 0 0 -> $2,500.00
    text = re.sub(r'\$\s*(\d+)\s*,\s*(\d+)\s*\.\s*(\d+)', r'$\1,\2.\3', text)
    text = re.sub(r'\$\s*(\d+)\s*,\s*(\d+)', r'$\1,\2', text)
    text = re.sub(r'\$\s*(\d+)\s*\.\s*(\d+)', r'$\1.\2', text)
    
    # Fix individual digits that got separated (like 5 0 0 -> 500)
    # Be careful not to break legitimate spaces
    text = re.sub(r'(\d)\s+(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', text)  # 4 digits
    text = re.sub(r'(\d)\s+(\d)\s+(\d)', r'\1\2\3', text)           # 3 digits
    text = re.sub(r'(\d)\s+(\d)(?=\s|$|[^\d])', r'\1\2', text)      # 2 digits
    
    return text

def extract_text_robust(filepath):
    """
    Enhanced text extraction with OpenAI GPT-4o Vision API as the primary method.
    Priority: 1) OpenAI GPT-4o Vision (highest accuracy), 2) PyMuPDF, 3) EasyOCR, 4) Tesseract
    """
    try:
        print(f"🚀 Starting enhanced text extraction for: {filepath}")
        
        # Method 1: OpenAI GPT-4o Vision API (Primary Method - Highest Accuracy)
        print("🤖 Attempting OpenAI GPT-4o Vision extraction (primary method)...")
        try:
            if openai_client and OPENAI_API_KEY:
                ai_text = extract_text_with_openai_vision(filepath)
                if len(ai_text.strip()) > 100:  # If we got substantial text
                    print(f"✅ GPT-4o Vision extraction successful: {len(ai_text)} characters")
                    
                    # Quality validation for contracts
                    word_count = len(ai_text.split())
                    has_contract_indicators = any(term in ai_text.lower() for term in [
                        'agreement', 'contract', 'party', 'parties', 'terms', 'conditions',
                        'signature', 'signed', 'whereas', 'hereby', 'thereof'
                    ])
                    
                    if word_count > 200 and has_contract_indicators:
                        print(f"🎯 High-quality contract text extracted with GPT-4o ({word_count} words)")
                        return ai_text
                    elif word_count > 50:
                        print(f"📄 Document text extracted with GPT-4o ({word_count} words)")
                        return ai_text
                    else:
                        print("⚠️ GPT-4o extraction returned minimal text, trying fallback methods...")
                else:
                    print("⚠️ GPT-4o extraction returned insufficient text, trying fallback methods...")
            else:
                print("⚠️ OpenAI API not configured, skipping GPT-4o extraction")
        except Exception as e:
            print(f"❌ GPT-4o extraction failed: {e}, falling back to traditional methods...")
        
        # Method 2: PyMuPDF (Fast, good for digital PDFs)
        print("📄 Trying PyMuPDF extraction...")
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            print(f"PyMuPDF extracted {len(text)} characters")
            
            # If we got substantial text from PyMuPDF, return it
            if len(text.strip()) > 100:
                print("✅ PyMuPDF extraction successful")
                return text
        except Exception as e:
            print(f"❌ PyMuPDF failed: {e}")
            text = ""
        
        # Method 3: EasyOCR (For scanned documents)
        print("📷 PyMuPDF found minimal text, trying EasyOCR...")
        try:
            ocr_text = extract_text_with_easyocr(filepath)
            if len(ocr_text.strip()) > 100:
                print(f"✅ EasyOCR extraction successful: {len(ocr_text)} characters")
                return ocr_text
        except Exception as e:
            print(f"❌ EasyOCR failed: {e}")
            ocr_text = ""
        
        # Method 4: Tesseract OCR (Last resort)
        print("🔍 Trying Tesseract OCR as final fallback...")
        try:
            images = convert_from_path(filepath, dpi=200)  # Lower DPI for speed
            tesseract_text = ""
            for img in images[:5]:  # Limit to 5 pages for speed
                tesseract_text += pytesseract.image_to_string(img)
            
            if len(tesseract_text.strip()) > 100:
                print(f"✅ Tesseract extraction successful: {len(tesseract_text)} characters")
                return tesseract_text
        except Exception as e:
            print(f"❌ Tesseract failed: {e}")
            tesseract_text = ""
        
        # Return the best result we have, even if minimal
        best_text = max([text, ocr_text, tesseract_text], key=len) if 'text' in locals() and 'ocr_text' in locals() and 'tesseract_text' in locals() else ""
        
        if best_text and len(best_text.strip()) > 0:
            print(f"⚠️ Returning best available extraction: {len(best_text)} characters")
            return best_text
        else:
            return "Error: Could not extract readable text from this document. The file may be corrupted, password-protected, or contain only images."
        
    except Exception as e:
        print(f"❌ Critical error in text extraction: {e}")
        return "Error: Unable to extract text from this document due to a processing error."

def extract_json_from_response(response_text):
    """Extract and parse JSON from model response, handling incomplete JSON"""
    try:
        # Try to find JSON pattern in the text
        import re
        import json
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1).strip()
            # Try parsing the extracted JSON
            return json.loads(json_str)
        else:
            # Fall back to looking for any JSON-like structure
            json_match = re.search(r'\{\s*"clauses"\s*:\s*\[[\s\S]*?\]\s*\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
            
            print("No JSON pattern found in response")
            return {"clauses": []}
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return {"clauses": []}

def analyze_contract(summary, original_text=None):
    """
    Enhanced contract analysis with precise source tracking.
    Uses improved prompt engineering to get exact text locations from DeepSeek.
    """
    # If in offline mode, return mock clauses
    if OFFLINE_MODE:
        print("Using offline mode - returning mock clauses")
        return MOCK_CLAUSES
    
    # Process the original text if provided
    if original_text:
        # Extract and number sentences for precise tracking
        sentences = extract_sentences(original_text)
        print(f"Extracted {len(sentences)} sentences for analysis")
        
        # Create numbered sentences for AI analysis
        numbered_sentences = []
        for i, sentence in enumerate(sentences, 1):
            numbered_sentences.append(f"[{i}] {sentence}")
        
        # Join sentences with clear separators
        sentences_text = "\n\n".join(numbered_sentences)
        
        # Enhanced system message for precise source tracking
        system_message = """You are an expert contract risk analyst. Your job is to identify business risks and provide EXACT SOURCE LOCATIONS.

CRITICAL REQUIREMENTS:
1. For each risk you identify, you MUST provide the exact sentence number where it appears
2. Use the [number] format from the text to reference source locations
3. Copy the EXACT text phrase that contains the risk (for highlighting)
4. Calculate precise 0-100 risk scores using the methodology below

RISK SCORING (0-100 Scale):
• 0-30: SAFE - Minimal impact, routine terms
• 31-69: WARNING - Moderate concern, needs attention  
• 70-100: UNSAFE - Critical threat, immediate action required

SCORING CRITERIA (weighted average):
1. Financial Impact (30%): 0-100 based on potential costs
2. Business Disruption (25%): 0-100 based on operational impact
3. Legal/Compliance Risk (20%): 0-100 based on legal exposure
4. Likelihood (15%): 0-100 based on probability of occurrence
5. Mitigation Difficulty (10%): 0-100 based on how hard to resolve

JSON RESPONSE FORMAT:
        ```json
        {
          "clauses": [
            {
              "id": 1,
      "sentence_numbers": [5, 6],
      "exact_text": "Late payment will incur 5% monthly penalty plus immediate acceleration",
      "type": "Payment Default Penalties", 
      "risk_score": 75,
      "risk_category": "Unsafe",
      "clause": "Business-friendly description of the risk",
      "consequences": "What could happen to the business",
      "amount": "Specific dollar amounts or percentages",
      "deadline": "Timeframes mentioned",
      "likelihood": "How likely to occur",
      "mitigation": "How to reduce this risk",
      "responsible_party": "Who is liable",
      "scoring_breakdown": {
        "financial_impact": 80,
        "business_disruption": 60,
        "legal_risk": 70,
        "likelihood": 85,
        "mitigation_difficulty": 90,
        "calculation": "80×0.3 + 60×0.25 + 70×0.2 + 85×0.15 + 90×0.1 = 75"
      }
    }
  ]
}"""

        prompt = f"""ANALYZE this contract and identify the TOP 8-10 MOST CRITICAL business risks.

MANDATORY REQUIREMENTS:
1. For each risk, provide the sentence number(s) where it appears using [number] format
2. Copy the EXACT TEXT phrase containing the risk (this will be highlighted)
3. Calculate precise risk scores (0-100) using the 5-criteria methodology
4. Focus on material business risks that could cost money or disrupt operations

CONTRACT TEXT (numbered sentences):
        
        {sentences_text}
        
ANALYSIS INSTRUCTIONS:
• Look for: payment penalties, termination risks, liability exposure, compliance requirements
• Ignore: routine boilerplate, standard legal language, administrative terms  
• For each risk: identify sentence numbers, copy exact risky text, calculate scores
• Provide actionable business insights, not legal analysis

Return JSON with precise source tracking for each identified risk."""

    else:
        # Fallback for when we only have summary (less precise but still functional)
        system_message = """You are a contract risk analyst. Identify the most critical business risks from this contract summary.

Focus on HIGH-IMPACT risks:
- Financial penalties and unexpected costs
- Termination conditions and business disruption
- Legal liability and compliance exposure
- Payment obligations and cash flow risks

Calculate 0-100 risk scores:
• 0-30: SAFE • 31-69: WARNING • 70-100: UNSAFE

JSON Format:
        ```json
        {
          "clauses": [
            {
              "id": 1,
      "type": "Risk Category",
      "risk_score": 75,
      "risk_category": "Unsafe", 
      "clause": "Business description",
      "consequences": "Potential impact",
      "amount": "Financial exposure",
      "deadline": "Timeframe",
      "likelihood": "Probability",
      "mitigation": "How to reduce risk",
      "responsible_party": "Who is liable"
    }
  ]
}"""

        prompt = f"""Analyze this contract summary and identify the TOP 10 MOST CRITICAL RISKS:

        {summary}
        
Focus on risks that could significantly impact the business financially or operationally.
Calculate precise 0-100 risk scores and provide specific financial amounts when available."""
    
    try:
        # Make API call to DeepSeek
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        print(f"DEEPSEEK RESPONSE LENGTH: {len(response_text)} characters")
        
        # Extract and parse JSON response
        result = extract_json_from_response(response_text)
        clauses = result.get("clauses", [])
        
        if not clauses:
            print("No clauses found in AI response")
            return []
        
        # Process clauses and map to original text
        processed_clauses = []
        
        for clause in clauses:
            # Handle risk scoring (convert to old format for compatibility)
            risk_score = clause.get('risk_score', 50)
            if risk_score <= 30:
                clause['risk'] = 'Low'
                clause['risk_category'] = 'Safe'
            elif risk_score <= 69:
                clause['risk'] = 'Medium'
                clause['risk_category'] = 'Warning'
            else:
                clause['risk'] = 'High'
                clause['risk_category'] = 'Unsafe'
            
            # Map back to original text using sentence numbers
            if original_text and 'sentence_numbers' in clause:
                sentence_numbers = clause['sentence_numbers']
                if isinstance(sentence_numbers, list) and sentence_numbers:
                    # Get the sentences referenced by the AI
                    referenced_sentences = []
                    for sent_num in sentence_numbers:
                        if isinstance(sent_num, int) and 1 <= sent_num <= len(sentences):
                            referenced_sentences.append(sentences[sent_num - 1])
                    
                    if referenced_sentences:
                        # Store the original text for highlighting
                        clause['original_text'] = ' '.join(referenced_sentences)
                        
                        # Find this text in the original document for offset calculation
                        combined_text = ' '.join(referenced_sentences)
                        start_offset = original_text.find(combined_text)
                        
                        if start_offset == -1:
                            # Try to find the exact_text instead
                            exact_text = clause.get('exact_text', '')
                            if exact_text:
                                start_offset = original_text.find(exact_text)
                                if start_offset != -1:
                                    combined_text = exact_text
                        
                        if start_offset != -1:
                            end_offset = start_offset + len(combined_text)
                            clause["textOffsets"] = {
                                "start": start_offset,
                                "end": end_offset
                            }
                            clause["highlight_text"] = combined_text
                            
                            # Add context snippet
                            context_start = max(0, start_offset - 50)
                            context_end = min(len(original_text), end_offset + 50)
                            clause["contextSnippet"] = original_text[context_start:context_end]
                    else:
                        print(f"Warning: Could not locate text for clause: {clause.get('type')}")
                        # Fallback: use the exact_text from AI
                        clause["highlight_text"] = clause.get('exact_text', clause.get('clause', ''))
                        clause["contextSnippet"] = clause.get('exact_text', clause.get('clause', ''))
            
            # Clean up amounts field
            amount = clause.get('amount', '')
            if amount in ["...", "Not specified", "Unknown", "", "N/A", None]:
                clause['amount'] = "Not specified"
            elif amount and not any(symbol in str(amount) for symbol in "$€£¥%"):
                # Add currency symbol if it looks like a monetary amount
                if re.search(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', str(amount)):
                    clause['amount'] = "$" + str(amount)
            
            # Clean up deadline field
            deadline = clause.get('deadline', '')
            if deadline in ["...", "Not specified", "Unknown", "", "N/A", None]:
                clause['deadline'] = "Not specified"
            
            processed_clauses.append(clause)
        
        # Sort by risk score (highest first)
        processed_clauses = sorted(processed_clauses, key=lambda x: x.get('risk_score', 50), reverse=True)
        
        print(f"Successfully processed {len(processed_clauses)} risk clauses with source tracking")
        return processed_clauses[:15]  # Limit to top 15 risks
        
    except Exception as e:
        print(f"Error processing contract analysis: {e}")
        return []

def extract_sentences(text):
    """
    Extract individual sentences from document text for analysis.
    This ensures we can highlight actual text that exists in the document.
    """
    import re
    
    # Clean the text - remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences (using common sentence-ending punctuation)
    # This regex handles common sentence endings while trying to avoid splitting
    # on periods in abbreviations, numbers, etc.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Further split long sentences into manageable chunks (for better highlighting)
    result = []
    for sentence in sentences:
        if len(sentence) > 200:  # If sentence is very long
            # Split on common delimiters within a sentence
            chunks = re.split(r'(?<=;|:|\n)\s+', sentence)
            result.extend(chunks)
        else:
            result.append(sentence)
    
    # Remove very short or empty sentences
    result = [s.strip() for s in result if len(s.strip()) > 10]
    
    return result

# Add these imports if they don't exist already at the top of the file
import asyncio
import aiohttp
import json
from functools import partial
import time

# Add this function after extract_monetary_values function
async def analyze_chunk_async(session, chunk, chunk_index, document_id, overlap_context=""):
    """
    Asynchronously analyze a single document chunk using API calls.
    
    Args:
        session: aiohttp ClientSession for making HTTP requests
        chunk: Text chunk to analyze
        chunk_index: Index of this chunk in the document
        document_id: ID of the document being processed
        overlap_context: Text from adjacent chunks for context preservation
    
    Returns:
        Analysis result for this chunk
    """
    try:
        # Add overlap context if available
        if overlap_context:
            enhanced_chunk = f"{overlap_context}\n\n{chunk}"
        else:
            enhanced_chunk = chunk
            
        # Extract sentences for this chunk
        chunk_sentences = extract_sentences(enhanced_chunk)
        
        # Format sentences for analysis
        sentences_text = ""
        for i, sent in enumerate(chunk_sentences):
            sentences_text += f"[{i+1}] {sent}\n\n"
        
        # Prepare the prompt for contract analysis
        prompt = f"""
        You are analyzing a contract to identify CRITICAL BUSINESS RISKS that require management attention.
        Think like a business advisor, not a lawyer. Focus on risks that could:
        
        1. COST MONEY: Financial penalties, unexpected costs, payment obligations
        2. DISRUPT OPERATIONS: Service interruptions, compliance burdens, resource constraints
        3. CREATE LIABILITY: Legal exposure, unlimited damages, indemnification
        4. RESTRICT GROWTH: Non-compete clauses, exclusivity restrictions, IP limitations
        5. THREATEN CONTINUITY: Termination risks, key person dependencies, renewal uncertainties
        
        ANALYZE THESE CONTRACT SENTENCES AND CONSOLIDATE RELATED RISKS:
        
        {sentences_text}
        
        INSTRUCTIONS:
        - Combine similar risks into single, comprehensive analyses
        - Use business-friendly language (avoid legal jargon)
        - Quantify financial exposure wherever possible
        - Provide specific mitigation recommendations
        - Focus on TOP 5-8 MOST MATERIAL RISKS only
        - Each risk should answer: "What could go wrong and what would it cost us?"
        
        EXAMPLES OF GOOD BUSINESS TITLES:
        ❌ Bad: "Indemnification Clause"
        ✅ Good: "Unlimited Legal Liability for Third-Party Claims"
        
        ❌ Bad: "Material Adverse Change"  
        ✅ Good: "Lender Can Demand Immediate Repayment During Market Downturns"
        
        ❌ Bad: "Termination for Convenience"
        ✅ Good: "Client Can Cancel Anytime, Leaving Us with Unrecovered Costs"
        
        Return comprehensive risk analysis focusing on business impact and actionability.
        """
        
        # Make API call to analyze this chunk
        # This example assumes you're using an API endpoint, replace with your actual API logic
        api_response = await analyze_contract_api(session, prompt, document_id, chunk_index)
        
        # If you're using your existing analyze_contract function:
        # api_response = await loop.run_in_executor(None, analyze_contract, sentences_text)
        
        # Enhance monetary values in results
        if 'clauses' in api_response:
            for clause in api_response['clauses']:
                # If amount not specified but exists in original text, try to extract
                if clause.get('amount') in ['N/A', 'Not specified', '...', ''] and clause.get('original_text'):
                    extracted_amounts = extract_monetary_values(clause['original_text'])
                    if extracted_amounts:
                        clause['amount'] = extracted_amounts[0]  # Use first found amount
        
        return api_response
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return {"error": str(e), "chunk_index": chunk_index}

# Replace the analyze_contract_api function with this implementation that uses your existing analyze_contract function
async def analyze_contract_api(session, prompt, document_id, chunk_index):
    """Make async API call to language model for contract analysis"""
    try:
        # Create a reference to your existing analyze_contract function
        loop = asyncio.get_running_loop()
        
        # Run your existing analyze_contract function in a thread pool executor
        # This allows the CPU-bound analyze_contract to run without blocking the event loop
        response = await loop.run_in_executor(None, analyze_contract, prompt)
        
        # If analyze_contract already returns a dict, return it directly
        if isinstance(response, dict) and 'clauses' in response:
            return response
            
        # If analyze_contract returns something else (like JSON text), process it
        if isinstance(response, str):
            return extract_json_from_response(response)
            
        # Handle other return types
        return {"error": "Unexpected response format", "chunk_index": chunk_index}
    except Exception as e:
        print(f"API error for chunk {chunk_index}: {str(e)}")
        return {"error": str(e), "chunk_index": chunk_index}

def handle_large_document(text, status_logs):
    """
    Args:
        text: The document text
        status_logs: List for status logging

    Returns:
        Analysis results
    """
    # Get document ID from status logs if available
    document_id = None
    for log in status_logs:
        if isinstance(log, dict) and 'document_id' in log:
            document_id = log['document_id']
            break

    if not document_id:
        status_logs.append("Error: No document ID found in status logs")
        # Fallback: Create a basic summary first, then analyze
        try:
            # Try to create a summary from the text first
            basic_summary = deepseek_summarize_full(text[:50000])  # Use first 50k chars for summary
            return analyze_contract(basic_summary, text)  # Correct parameters: summary, original_text
        except Exception as e:
            status_logs.append(f"Fallback summary failed: {str(e)}")
            # Last resort: return mock results
            return {"clauses": []}

    try:
        # Create and run event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use the extractive summarization approach instead of async processing
            status_logs.append("Processing large document with extractive summarization")
            
            # Calculate target sentences based on document size
            estimated_sentences = len(text) // 80  # Rough estimate: 80 chars per sentence
            target_sentences = min(50, max(20, estimated_sentences // 50))  # 20-50 sentences
            
            status_logs.append(f"Creating extractive summary with {target_sentences} key sentences")
            extractive_summary = extractive_summarize_large_document(text, target_sentences)
            
            # Create AI summary from the extractive summary
            status_logs.append("Generating final summary from key sentences")
            summary = deepseek_summarize_full(extractive_summary)
            
            # Analyze the contract using the summaries
            status_logs.append("Analyzing contract for risks")
            clauses = analyze_contract(summary, extractive_summary)
            
            status_logs.append("Large document processing complete")
            return {"clauses": clauses}
        finally:
            loop.close()
    except Exception as e:
        error_msg = f"Error in parallel document processing: {str(e)}"
        status_logs.append(error_msg)
        print(error_msg)
        # Fall back to direct analysis
        return analyze_contract(text)

# Modify the process_document function to use handle_large_document for all documents
def process_document(filepath, document_id):
    """Process document in background with status updates"""
    try:
        # Update status to extracting
        document_processing_status[document_id]['status'] = 'extracting'
        document_processing_status[document_id]['current_step'] = 'Extracting text from PDF...'
        document_processing_status[document_id]['progress'] = 10
        
        # Extract text
        text = extract_text_robust(filepath)
        
        # Preprocess the text for better analysis
        clean_text = preprocess_text(text)
        
        # Update status to summarizing
        document_processing_status[document_id]['status'] = 'summarizing'
        document_processing_status[document_id]['current_step'] = 'Generating contract summary...'
        document_processing_status[document_id]['progress'] = 40
        
        # Generate summary
        status_logs = []  # Initialize status logs
        # Add document_id to status_logs so handle_large_document can find it
        status_logs.append({"document_id": document_id, "message": "Starting large document processing"})
        
        if len(clean_text) > 240000:
            try:
                # Use the extractive summarization approach for large documents
                status_logs.append("Processing large document with extractive summarization")
                
                # Calculate target sentences based on document size
                estimated_sentences = len(text) // 80  # Rough estimate: 80 chars per sentence
                target_sentences = min(50, max(20, estimated_sentences // 50))  # 20-50 sentences
                
                status_logs.append(f"Creating extractive summary with {target_sentences} key sentences")
                extractive_summary = extractive_summarize_large_document(text, target_sentences)
                
                # Create AI summary from the extractive summary
                status_logs.append("Generating final summary from key sentences")
                summary = deepseek_summarize_full(extractive_summary)
                
                # Analyze the contract using the summaries
                status_logs.append("Analyzing contract for risks")
                clauses = analyze_contract(summary, extractive_summary)
                
                status_logs.append("Large document processing complete")
                return {"clauses": clauses}
                
            except Exception as e:
                error_msg = f"Error in large document processing: {str(e)}"
                status_logs.append(error_msg)
                print(error_msg)
                # Fall back to basic analysis with truncated text
                try:
                    truncated_text = text[:50000] + "... [Document truncated for processing]"
                    summary = deepseek_summarize_full(truncated_text)
                    clauses = analyze_contract(summary)
                    return {"clauses": clauses}
                except Exception as e2:
                    status_logs.append(f"Fallback analysis also failed: {str(e2)}")
                    return {"clauses": []}
        else:
            summary = deepseek_summarize_full(clean_text)
        
        # Update status to analyzing
        document_processing_status[document_id]['status'] = 'analyzing'
        document_processing_status[document_id]['current_step'] = 'Identifying risk factors...'
        document_processing_status[document_id]['progress'] = 70
        
        # Extract sentences for better highlighting
        sentences = extract_sentences(clean_text)
        status_logs.append(f"Extracted {len(sentences)} sentences for risk analysis")
        
        # Analyze contract for risks, using appropriate text size
        try:
            if len(clean_text) > 240000:
                # For large documents, use extractive summary for risk analysis too
                status_logs.append("Using extractive summary for risk analysis")
                extractive_summary_for_analysis = extractive_summarize_large_document(clean_text, target_sentences=30)
                clauses = analyze_contract(summary, extractive_summary_for_analysis)
            else:
                # For normal documents, use full text
                clauses = analyze_contract(summary, clean_text)
        except Exception as e:
            status_logs.append(f"Risk analysis failed: {str(e)}")
            # Fallback to basic analysis without original text
            try:
                clauses = analyze_contract(summary)
            except Exception as e2:
                status_logs.append(f"Basic analysis also failed: {str(e2)}")
                clauses = []  # Empty results as last resort
        
        # Sort clauses by risk level
        clauses = sorted(clauses, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x["risk"], 3))
        
        # Enhanced clause matching - store the exact text from the document for highlighting
        for clause in clauses:
            # First check if we have exact_text from the AI model
            if "exact_text" in clause and clause["exact_text"]:
                highlight_text = clause["exact_text"]
            # Check if we already have original_text from sentence-based analysis
            elif "original_text" in clause and clause["original_text"]:
                highlight_text = clause["original_text"]
                # Use the clause description as fallback
            else:
                highlight_text = clause.get("clause", "")
            
            # Advanced text matching using enhanced algorithm for layout-preserved text
            # 1. First try exact match with enhanced algorithm
            start_offset = enhanced_text_matching(clean_text, highlight_text)
            
            # 2. If enhanced matching fails, try traditional approaches
            if start_offset == -1:
                # Try exact traditional match
                start_offset = clean_text.find(highlight_text)
            
            # 3. If exact match fails, try case-insensitive match
            if start_offset == -1:
                lower_text = clean_text.lower()
                lower_highlight = highlight_text.lower()
                start_offset = lower_text.find(lower_highlight)
                if start_offset != -1:
                    # If found, use the actual text at this position (preserving case)
                    highlight_text = clean_text[start_offset:start_offset + len(highlight_text)]
            
            # 4. If that fails, try with smaller segments - first check type field
            if start_offset == -1 and "type" in clause:
                clause_type = clause.get("type")
                if clause_type and len(clause_type) > 3:  # Only use meaningful types
                    type_offset = clean_text.lower().find(clause_type.lower())
                    if type_offset != -1:
                        # Found the clause type, get surrounding context
                        context_start = max(0, type_offset - 20)
                        context_end = min(len(clean_text), type_offset + len(clause_type) + 100)
                        highlight_text = clean_text[context_start:context_end]
                        start_offset = context_start
            
            # 5. Try with key words from the clause text if still no match
            if start_offset == -1:
                # Get significant words from highlight text
                import re
                words = re.findall(r'\b\w{4,}\b', highlight_text.lower())
                if words:
                    # Look for clusters of significant words
                    matches = []
                    for word in words:
                        word_offset = clean_text.lower().find(word)
                        if word_offset != -1:
                            matches.append((word_offset, word))
                    
                    if matches:
                        # Sort matches by position
                        matches.sort(key=lambda x: x[0])
                        # Find clusters (words close to each other)
                        if len(matches) > 1:
                            for i in range(len(matches) - 1):
                                current_pos = matches[i][0]
                                next_pos = matches[i+1][0]
                                # If words are close (within 100 chars)
                                if next_pos - current_pos < 100:
                                    context_start = max(0, current_pos - 10)
                                    context_end = min(len(clean_text), next_pos + len(matches[i+1][1]) + 10)
                                    highlight_text = clean_text[context_start:context_end]
                                    start_offset = context_start
                                    break
                        # If no clusters or just one word, use first significant word with context
                        if start_offset == -1 and matches:
                            first_match = matches[0]
                            context_start = max(0, first_match[0] - 10)
                            context_end = min(len(clean_text), first_match[0] + len(first_match[1]) + 50)
                            highlight_text = clean_text[context_start:context_end]
                            start_offset = context_start
            
            # If any matching approach succeeded
            if start_offset != -1:
                end_offset = start_offset + len(highlight_text)
                # Add offsets to clause data
                clause["textOffsets"] = {
                    "start": start_offset,
                    "end": end_offset,
                }
                
                # Also store a context snippet for highlighting
                context_start = max(0, start_offset - 50)
                context_end = min(len(clean_text), end_offset + 50)
                clause["contextSnippet"] = clean_text[context_start:context_end]
                
                # Store the highlight text separately
                clause["highlight_text"] = highlight_text
            else:
                # If all matching approaches failed, log it
                print(f"Warning: Could not find a match for clause: {clause.get('type', 'Unknown type')}")
                # Still store a default context to ensure something appears
                if "type" in clause:
                    clause["contextSnippet"] = f"{clause.get('type')}: {clause.get('clause', '')}"
                    clause["highlight_text"] = clause.get('clause', '')
        
        # Preview text for display
        preview_text = clean_text[:1000] + "..." if len(clean_text) > 1000 else clean_text
        
        # Update status to complete
        document_processing_status[document_id]['status'] = 'complete'
        document_processing_status[document_id]['current_step'] = 'Analysis complete!'
        document_processing_status[document_id]['progress'] = 100
        document_processing_status[document_id]['complete'] = True
        document_processing_status[document_id]['preview_text'] = preview_text
        document_processing_status[document_id]['summary'] = summary
        document_processing_status[document_id]['clauses'] = clauses
        document_processing_status[document_id]['processing_time'] = int(time.time() - document_processing_status[document_id]['start_time'])
        document_processing_status[document_id]['status_logs'] = status_logs
        document_processing_status[document_id]['pdf_path'] = filepath
        document_processing_status[document_id]['original_text'] = clean_text  # Ensure we store the complete text for later use
        
        # Store results (you'd implement caching here)
        print(f"Processing complete: {filepath}")
    except Exception as e:
        # Update status to error
        document_processing_status[document_id]['status'] = 'error'
        document_processing_status[document_id]['current_step'] = f'Error: {str(e)}'
        document_processing_status[document_id]['complete'] = True
        document_processing_status[document_id]['error'] = str(e)
        print(f"Error processing document: {e}")

def preprocess_text(text):
    """
    Clean up extracted text to make it more suitable for LLM processing.
    """
    import re
    
    # Remove unprintable characters
    text = ''.join(c if c.isprintable() or c in '\n\t' else ' ' for c in text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove repeated patterns that might be garbage
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
    
    # Remove very long tokens (likely garbage)
    text = re.sub(r'\S{50,}', '[UNREADABLE]', text)
    
    return text.strip()

def truncate_for_model(text, max_tokens=30000):
    """Intelligently truncate text to fit within token limits while preserving context."""
    # Count tokens (approximate)
    words = text.split()
    estimated_tokens = len(words)
    
    # If already under limit, return as is
    if estimated_tokens <= max_tokens:
        return text
    
    # Try to split by sections first
    sections = re.split(r'(?:\n\s*){2,}(?:[A-Z\s]{5,}:|\d+\.\s+[A-Z])', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for section in sections:
        section_tokens = len(section.split())
        
        # If this section alone exceeds limit, split it further
        if section_tokens > max_tokens:
            # Split by paragraphs
            paragraphs = section.split('\n\n')
            for para in paragraphs:
                para_tokens = len(para.split())
                if para_tokens + current_tokens <= max_tokens:
                    current_chunk += para + '\n\n'
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # If paragraph still too big, split by sentences
                    if para_tokens > max_tokens:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        current_chunk = ""
                        current_tokens = 0
                        for sent in sentences:
                            sent_tokens = len(sent.split())
                            if sent_tokens + current_tokens <= max_tokens:
                                current_chunk += sent + ' '
                                current_tokens += sent_tokens
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = sent + ' '
                                current_tokens = sent_tokens
                    else:
                        current_chunk = para + '\n\n'
                        current_tokens = para_tokens
        # If adding this section would exceed limit, start new chunk
        elif current_tokens + section_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = section
            current_tokens = section_tokens
        # Otherwise add to current chunk
        else:
            current_chunk += section
            current_tokens += section_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we ended up with only one chunk, fall back to simpler method
    if len(chunks) <= 1:
        # Simple truncation as fallback
        return ' '.join(words[:max_tokens])
    
    return chunks

def handle_large_document(text, status_logs):
    """
    Process very large documents by chunking and hierarchical summarization.
    This allows us to handle contracts of virtually any length.
    """
    overall_start_time = time.time()
    process_id = str(uuid.uuid4())
    
    # Get dynamic parameters based on document size
    params = calculate_dynamic_parameters(len(text))
    
    # Split into reasonably sized chunks
    section_id = f"{process_id}-section"
    split_start_time = time.time()
    
    # Add logging to status_logs array instead of using SSE
    log_message = "Splitting document into sections..."
    status_logs.append(log_message)
    
    # More conservative initial chunk size (start smaller than we might need)
    initial_chunk_size = min(params['chunk_size'], 40000)  # Start with smaller chunks by default
    
    # First, estimate token count for the whole document
    estimated_tokens = len(text) / 4  # Rough approximation: 4 chars = 1 token
    
    # Calculate how many chunks we need to stay under token limits
    max_tokens_per_chunk = 60000  # Keep well under DeepSeek's 65k limit
    min_chunks_needed = math.ceil(estimated_tokens / max_tokens_per_chunk)
    
    # Adjust chunk size to ensure we have at least the minimum number of chunks
    adjusted_chunk_size = min(initial_chunk_size, len(text) // min_chunks_needed)
    
    status_logs.append(f"Estimated tokens: {estimated_tokens:.0f}, using conservative chunk size")
    log_message = f"Estimated {estimated_tokens:.0f} tokens, dividing document appropriately..."
    status_logs.append(log_message)
    
    chunks = [text[i:i+adjusted_chunk_size] for i in range(0, len(text), adjusted_chunk_size)]
    status_logs.append(f"Split into {len(chunks)} sections")
    
    log_message = f"Document split into {len(chunks)} sections for processing"
    status_logs.append(log_message)
    
    # Summarize each chunk separately
    section_summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"{process_id}-chunk-{i}"
        chunk_start_time = time.time()
        
        log_message = f"Processing section {i}/{len(chunks)}..."
        status_logs.append(log_message)
        
        try:
            # Add a pre-check for token count
            chunk_tokens = len(chunk) / 4
            if chunk_tokens > 60000:  # If still too large
                log_message = f"Section {i} is large ({chunk_tokens:.0f} tokens), subdividing..."
                status_logs.append(log_message)
                
                # Sub-divide this chunk
                sub_chunks = [chunk[j:j+25000] for j in range(0, len(chunk), 25000)]
                sub_summaries = []
                for j, sub_chunk in enumerate(sub_chunks, 1):
                    sub_id = f"{chunk_id}-sub-{j}"
                    sub_start_time = time.time()
                    
                    log_message = f"Processing sub-section {i}.{j}..."
                    status_logs.append(log_message)
                    
                    try:
                        sub_summary = deepseek_summarize_chunk(sub_chunk, params['sentence_count'])
                        sub_summaries.append(sub_summary)
                        
                        log_message = f"Sub-section {i}.{j} processed successfully"
                        status_logs.append(log_message)
                    except Exception as e:
                        log_message = f"Error in sub-section {i}.{j}: {str(e)}"
                        status_logs.append(log_message)
                
                # Combine sub-summaries
                summary = " ".join(sub_summaries)
            else:
                # Process normally
                summary = deepseek_summarize_chunk(chunk, params['sentence_count'])
            
            section_summaries.append(summary)
            log_message = f"Section {i}/{len(chunks)} processed successfully"
            status_logs.append(log_message)
            
        except Exception as e:
            log_message = f"Error in section {i}, trying with smaller chunk"
            status_logs.append(log_message)
            
            # Try again with a much smaller chunk size
            retry_id = f"{chunk_id}-retry"
            retry_start_time = time.time()
            
            log_message = f"Retrying section {i} with reduced size..."
            status_logs.append(log_message)
            
            try:
                # Cut the chunk in half
                half_chunk = chunk[:len(chunk)//2]
                half_summary = deepseek_summarize_chunk(half_chunk, params['sentence_count'])
                section_summaries.append(half_summary + " [Note: This is a partial section summary due to size constraints.]")
                
                log_message = f"Section {i} retry succeeded with reduced size"
                status_logs.append(log_message)
            except Exception as e2:
                log_message = f"Second attempt failed: {str(e2)}"
                status_logs.append(log_message)
                section_summaries.append(f"[Section {i} summary failed]")
    
    # Combine all section summaries
    combine_id = f"{process_id}-combine"
    combine_start_time = time.time()
    
    log_message = "Combining section summaries..."
    status_logs.append(log_message)
    
    combined_text = "\n\n".join([f"SECTION {i+1}: {summary}" 
                                for i, summary in enumerate(section_summaries)])
    
    log_message = "All sections combined successfully"
    status_logs.append(log_message)
    
    # Create a final meta-summary
    final_id = f"{process_id}-final"
    final_start_time = time.time()
    
    log_message = "Creating final summary..."
    status_logs.append(log_message)
    
    result = deepseek_summarize_combined(combined_text, params['final_sentence_count'])
    
    log_message = "Final summary created successfully"
    status_logs.append(log_message)
    
    # Complete overall process
    log_message = "Document processing complete!"
    status_logs.append(log_message)
    
    return result

def deepseek_summarize_chunk(chunk, sentence_count):
    """Summarize a single chunk of a large document"""
    # If in offline mode, return a simpler summary
    if OFFLINE_MODE:
        return f"Mock summary of document section (offline mode). This would contain about {sentence_count} sentences summarizing the content."
        
    resp = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"Summarize this document section in {sentence_count} sentences."},
            {"role": "user", "content": f"Here is a section of a contract. Summarize the key points in {sentence_count} sentences:\n\n{chunk}"}
        ],
        temperature=0.2,
        max_tokens=250
    )
    return resp.choices[0].message.content.strip()

def deepseek_summarize_combined(text, sentence_count):
    """Create a final summary from all the section summaries"""
    # If in offline mode, return the mock summary
    if OFFLINE_MODE:
        return MOCK_SUMMARY
        
    resp = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"You are an expert contract summarizer. Create a coherent {sentence_count} sentence summary."},
            {"role": "user", "content": f"Below are summaries of different sections of a large contract. Create a coherent overall summary in {sentence_count} sentences:\n\n{text}"}
        ],
        temperature=0.2,
        max_tokens=500  # Allowing more tokens for very large contracts
    )
    return resp.choices[0].message.content.strip()

def calculate_dynamic_parameters(text_length):
    """
    Dynamically calculate summary parameters based on document length.
    Returns a tuple of (sentence_count, max_tokens, chunk_size)
    """
    # Scale summary length based on document size
    if text_length < 10000:  # Small contract (~2-3 pages)
        return {
            "sentence_count": "2-3",
            "max_tokens": 200,
            "chunk_size": 20000,  # Not used for small docs, but included for completeness
            "final_sentence_count": "3-4"  # For meta-summary
        }
    elif text_length < 50000:  # Medium contract (~10-15 pages)
        return {
            "sentence_count": "3-5",
            "max_tokens": 300,
            "chunk_size": 40000,
            "final_sentence_count": "4-6"
        }
    elif text_length < 200000:  # Large contract (~40-50 pages)
        return {
            "sentence_count": "5-7",
            "max_tokens": 400,
            "chunk_size": 60000,
            "final_sentence_count": "6-8"
        }
    else:  # Very large contract (50+ pages)
        return {
            "sentence_count": "6-8", 
            "max_tokens": 500,
            "chunk_size": 80000,
            "final_sentence_count": "8-12"
        }

def extractive_summarize_large_document(text, target_sentences=20):
    """
    Create an extractive summary of a large document using frequency-based sentence ranking.
    This works locally without API calls and handles documents of any size.
    """
    import re
    
    # Clean and split into sentences
    sentences = extract_sentences(text)
    if len(sentences) <= target_sentences:
        return " ".join(sentences)
    
    print(f"Extractive summarization: Processing {len(sentences)} sentences to find top {target_sentences}")
    
    # Tokenize and count word frequencies (exclude common stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'not', 'no'
    }
    
    # Count word frequencies across all sentences
    word_freq = Counter()
    for sentence in sentences:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())  # Extract words 3+ chars
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
    
    # Score each sentence based on word frequencies
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        score = 0
        word_count = 0
        
        # Boost scores for sentences with financial/legal terms
        high_value_terms = [
            'payment', 'penalty', 'fee', 'cost', 'price', 'amount', 'dollar', 'liability', 
            'indemnif', 'terminat', 'breach', 'default', 'deadline', 'notice', 'rent',
            'deposit', 'insurance', 'compliance', 'violation', 'damages', 'interest'
        ]
        
        for word in words:
            if word not in stop_words:
                score += word_freq[word]
                word_count += 1
                
                # Boost for important contract terms
                if any(term in word for term in high_value_terms):
                    score += word_freq[word] * 2  # Double weight for important terms
        
        # Normalize by sentence length and add position bonus (earlier sentences often important)
        if word_count > 0:
            normalized_score = score / word_count
            position_bonus = max(0, (len(sentences) - i) / len(sentences) * 0.1)  # Small bonus for earlier sentences
            final_score = normalized_score + position_bonus
        else:
            final_score = 0
            
        sentence_scores.append((final_score, i, sentence))
    
    # Sort by score and take top sentences
    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_sentences = sentence_scores[:target_sentences]
    
    # Sort selected sentences by original order to maintain document flow
    top_sentences.sort(key=lambda x: x[1])
    
    # Create summary
    summary_sentences = [sentence for _, _, sentence in top_sentences]
    summary = " ".join(summary_sentences)
    
    print(f"Extractive summary created: {len(summary)} characters from {len(sentences)} sentences")
    return summary

def deepseek_summarize_full(text: str) -> str:
    """
    Ask DeepSeek-Chat to summarize the entire contract with dynamic length.
    """
    # If in offline mode, return mock summary
    if OFFLINE_MODE:
        print("Using offline mode - returning mock summary")
        return MOCK_SUMMARY
        
    # If text is too short, return a message
    if len(text.strip()) < 50:
        return "The document appears to contain insufficient readable text for summarization."
    
    # Dynamically set parameters based on document length
    params = calculate_dynamic_parameters(len(text))
    
    # Add a retry mechanism with exponential backoff
    max_retries = 3
    retry_delay = 2  # Start with 2 second delay
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to summarize document (attempt {attempt + 1}/{max_retries})")
            
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert contract summarizer. Produce a concise {params['sentence_count']} sentence summary."
                    },
                    {
                        "role": "user",
                        "content": f"Here is the contract text. Summarize it in {params['sentence_count']} sentences, capturing the most important terms, conditions, and risks:\n\n{text}"
                    }
                ],
                temperature=0.2,
                max_tokens=params['max_tokens'],
                timeout=30.0  # Add 30-second timeout
            )
            
            result = resp.choices[0].message.content.strip()
            print(f"Successfully generated summary ({len(result)} characters)")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"API call failed (attempt {attempt + 1}): {e}")
            
            if "rate limit" in error_msg:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return "Unable to analyze contract due to API rate limits. Please try again later."
            elif "context length" in error_msg or "too many tokens" in error_msg:
                # Try with shorter text if we get context length errors
                if attempt < max_retries - 1:
                    print(f"Context length error, shortening text...")
                    text = truncate_for_model(text, max_tokens=15000)
                else:
                    return "Contract is too large for full analysis. Please use a smaller document."
            elif "timeout" in error_msg or "timed out" in error_msg:
                if attempt < max_retries - 1:
                    print(f"API timeout, retrying with longer timeout...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return "API timeout occurred. The service may be overloaded. Please try again later."
            elif attempt < max_retries - 1:
                print(f"API error: {e}, retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to analyze after {max_retries} attempts: {e}")
                return "An error occurred during contract analysis. Please try uploading the document again."

@app.route("/refresh_analysis", methods=["POST"])
def refresh_analysis():
    """Force a fresh analysis of the current document"""
    filepath = request.form.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return {"error": "Document not found"}, 400
    
    # Extract text from the document
    text = extract_text_robust(filepath)
    
    # Get a fresh summary
    if len(text) > 240000:
        summary = handle_large_document(text, [])
    else:
        summary = deepseek_summarize_full(text)
    
    # Get fresh clause analysis
    clauses = analyze_contract(summary, text)
    clauses = sorted(clauses, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x["risk"], 3))
    
    # Update the timestamp
    global current_analysis_time
    current_analysis_time = time.time()
    
    return {"summary": summary, "clauses": clauses}

@app.route('/preview/<filename>')
def preview(filename):
    # This route is for direct access to preview page
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found')
        return redirect(url_for('upload'))
        
    text = extract_text_robust(filepath)
    preview_text = text[:1000] + "..." if len(text) > 1000 else text
    
    # Use cached summary and analysis if available
    # You'd need to implement caching of results
    
    return render_template(
        "preview.html", 
        preview=preview_text, 
        filename=filename,
        summary="Summary not available - please upload the document again.",
        clauses=[],
        status_logs=[]
    )

@app.route('/check_status/<document_id>')
@login_required
def check_status(document_id):
    """Check the status of document processing"""
    if not is_valid_uuid(document_id):
        return jsonify({'error': 'Invalid document ID format'}), 400
        
    if document_id not in document_processing_status:
        return jsonify({'error': 'Document not found'}), 404
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    return jsonify({
        'status': status['status'],
        'progress': status['progress'],
        'current_step': status['current_step'],
        'complete': status['complete']
    })

@app.route('/results/<document_id>')
@login_required
def results(document_id):
    """Show processing results"""
    if not is_valid_uuid(document_id):
        flash('Invalid document ID format')
        return redirect(url_for('upload'))
        
    if document_id not in document_processing_status:
        flash('Document not found')
        return redirect(url_for('upload'))
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    if not status.get('complete'):
        # If not complete, redirect to analyzing page
        return redirect(url_for('analyzing', document_id=document_id))
    
    if status.get('status') == 'error':
        flash(f'Error processing document: {status.get("error")}')
        return redirect(url_for('upload'))
    
    # Calculate risk counts for the charts
    clauses = status.get('clauses', [])
    risk_counts = {
        "High": 0,
        "Medium": 0,
        "Low": 0
    }
    
    # Count risks by category
    for clause in clauses:
        risk = clause.get('risk')
        if risk in risk_counts:
            risk_counts[risk] += 1
    
    # Return the preview page with results
    return render_template(
        "preview.html", 
        preview=status.get('preview_text', ''),
        filename=status.get('filename', ''),
        filepath=status.get('filepath', ''),
        summary=status.get('summary', ''),
        clauses=status.get('clauses', []),
        processing_time=status.get('processing_time', 0),
        status_logs=status.get('status_logs', []),
        risk_counts=risk_counts,
        document_id=document_id
    )

@app.route('/analyzing/<document_id>')
@login_required
def analyzing(document_id):
    """Show analyzing page with specific document ID"""
    if not is_valid_uuid(document_id):
        flash('Invalid document ID format')
        return redirect(url_for('upload'))
        
    if document_id not in document_processing_status:
        flash('Document not found')
        return redirect(url_for('upload'))
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    # If already complete, redirect to results
    if status.get('complete'):
        return redirect(url_for('results', document_id=document_id))
    
    return render_template("analyzing.html", 
                          filename=status.get('filename', ''),
                          document_id=document_id)

@app.route('/delete_file', methods=["POST"])
@login_required
def delete_file():
    """Delete a file from the uploads folder"""
    filepath = request.form.get("filepath")
    
    # Validate the filepath is in the uploads folder (security check)
    if not filepath or not os.path.exists(filepath) or not filepath.startswith(app.config['UPLOAD_FOLDER']):
        return jsonify({"success": False, "message": "File not found or invalid path"}), 400
    
    # Verify user owns this file
    filename = os.path.basename(filepath)
    is_owner = False
    doc_id_to_delete = None
    
    for doc_id, status in document_processing_status.items():
        if status.get('filepath') == filepath:
            if status.get('user_id') != current_user.id:
                return jsonify({"success": False, "message": "Unauthorized access"}), 403
            is_owner = True
            doc_id_to_delete = doc_id
            break
    
    if not is_owner:
        return jsonify({"success": False, "message": "Unauthorized access"}), 403
    
    try:
        # Delete the file
        os.remove(filepath)
        
        # Also remove associated processing data
        if doc_id_to_delete:
            del document_processing_status[doc_id_to_delete]
        
        return jsonify({"success": True, "message": "File deleted successfully"}), 200
    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({"success": False, "message": f"Error deleting file: {str(e)}"}), 500

# Add this function for maintenance purposes
def cleanup_old_documents():
    """Clean up documents older than 7 days"""
    current_time = time.time()
    documents_to_delete = []
    
    # Find documents older than 7 days
    for doc_id, status in document_processing_status.items():
        document_age = current_time - status.get('start_time', current_time)
        if document_age > 7 * 24 * 60 * 60:  # 7 days in seconds
            documents_to_delete.append(doc_id)
            
            # Delete the file if it exists
            filepath = status.get('filepath')
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Deleted old file: {filepath}")
                except Exception as e:
                    print(f"Error deleting old file {filepath}: {e}")
    
    # Remove document data
    for doc_id in documents_to_delete:
        try:
            del document_processing_status[doc_id]
            print(f"Removed old document data: {doc_id}")
        except Exception as e:
            print(f"Error removing document data {doc_id}: {e}")
    
    return len(documents_to_delete)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view_pdf/<document_id>')
@login_required
def view_pdf(document_id):
    """Special PDF viewer with highlighting capability"""
    if not is_valid_uuid(document_id):
        flash('Invalid document ID format')
        return redirect(url_for('upload'))
        
    if document_id not in document_processing_status:
        flash('Document not found')
        return redirect(url_for('upload'))
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    if not status.get('complete'):
        # If not complete, redirect to analyzing page
        return redirect(url_for('analyzing', document_id=document_id))
        
    # Load file path - use the correct filepath from the status object
    filepath = status.get('filepath')
    if not filepath or not os.path.exists(filepath):
        flash('PDF file not found')
        return redirect(url_for('dashboard'))
        
    # Load extracted text
    original_text = status.get('original_text', '')
    if not original_text:
        # If original text wasn't stored in status, extract it now
        try:
            original_text = extract_text_robust(filepath)
        except Exception as e:
            print(f"Error extracting text: {e}")
            original_text = "Error extracting text from PDF."
    
    filename = status.get('filename', 'document.pdf')
    
    # Get the clauses for highlighting
    clauses = status.get('clauses', [])
    
    # Filter out any None or undefined clauses first
    clean_clauses = []
    for clause in clauses:
        if clause and isinstance(clause, dict):
            clean_clauses.append(clause)
    
    # Build highlight_terms array and highlight_data with safe serialization
    highlight_terms = []
    highlight_data = []
    
    for clause in clean_clauses[:10]:  # Limit to first 10 clauses to avoid URL length issues
        risk_level = clause.get('risk', 'Low')
        if risk_level:
            risk_level = risk_level.lower()
        else:
            risk_level = 'low'
        
        # Get the best text for highlighting with safe extraction
        highlight_text = ''
        
        # Try each field safely
        for field in ['contextSnippet', 'highlight_text', 'exact_text', 'original_text', 'clause']:
            value = clause.get(field)
            # Handle jinja2.Undefined objects
            if isinstance(value, jinja2.Undefined):
                continue
            if value and isinstance(value, str) and value.strip():
                highlight_text = value
                break
        
        # Only add if we have valid text
        if highlight_text and highlight_text.strip():
            # Clean the highlight text to ensure it's serializable
            clean_text = str(highlight_text).strip()[:200]  # Limit length and ensure it's a string
            highlight_terms.append(clean_text)
            
            # Build highlight data with safe field extraction
            def safe_get(clause_dict, key, default='Not specified'):
                value = clause_dict.get(key, default)
                # Handle jinja2.Undefined objects (similar to dbt-core issue CT-2259)
                if isinstance(value, jinja2.Undefined):
                    return default
                # Handle undefined/None values
                if value is None or str(value).lower() in ['undefined', 'none', '']:
                    return default
                return str(value)
            
            highlight_data.append({
                'text': clean_text,
                'risk': risk_level,
                'type': safe_get(clause, 'type', 'Risk Clause'),
                'consequences': safe_get(clause, 'consequences'),
                'amount': safe_get(clause, 'amount'),
                'deadline': safe_get(clause, 'deadline')
            })
    
    # Generate PDF URL path for the iframe
    pdf_url = url_for('uploaded_file', filename=filename)
    
    # Pass the necessary data to the template
    return render_template(
        'pdf_viewer.html', 
        document_id=document_id,
        filename=filename,
        text=original_text,
        clauses=clean_clauses,
        highlight_data=highlight_data,
        highlight_terms=highlight_terms,  # Add the missing variable
        pdf_url=pdf_url
    )

@app.route('/get_highlight_data/<document_id>', methods=['POST'])
@login_required
def get_highlight_data(document_id):
    """Get full highlight data via POST to avoid URL length limits"""
    if not is_valid_uuid(document_id):
        return jsonify({'error': 'Invalid document ID format'}), 400
        
    if document_id not in document_processing_status:
        return jsonify({'error': 'Document not found'}), 404
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    clauses = status.get('clauses', [])
    
    # Build complete highlight data
    highlight_data = []
    highlight_terms = []
    
    for clause in clauses:
        risk_level = clause.get('risk', 'Low').lower()
        
        # Get the best text for highlighting
        highlight_text = (
            clause.get('contextSnippet') or 
            clause.get('highlight_text') or 
            clause.get('exact_text') or 
            clause.get('original_text') or 
            clause.get('clause', '')
        )
        
        if highlight_text.strip():
            highlight_terms.append(highlight_text)
            highlight_data.append({
                'text': highlight_text,
                'risk': risk_level,
                'type': clause.get('type', 'Risk Clause'),
                'consequences': clause.get('consequences', 'Not specified'),
                'amount': clause.get('amount', 'Not specified'),
                'deadline': clause.get('deadline', 'Not specified')
            })
    
    return jsonify({
        'highlight_terms': highlight_terms,
        'highlight_data': highlight_data
    })

@app.route('/get_document_text/<document_id>')
def get_document_text(document_id):
    try:
        # Get document info from document_processing_status instead of MongoDB
        if document_id not in document_processing_status:
            return jsonify({"success": False, "error": "Document not found"})
        
        doc = document_processing_status[document_id]
        pdf_filepath = doc.get('filepath')
        
        if not pdf_filepath or not os.path.exists(pdf_filepath):
            return jsonify({"success": False, "error": "PDF file not found"})
        
        # Extract text from the PDF using our robust method
        text = extract_text_robust(pdf_filepath)
        
        if not text or text.strip() == "":
            return jsonify({"success": False, "error": "No text could be extracted from this PDF"})
            
        return jsonify({"success": True, "text": text})
    except Exception as e:
        print(f"Error in get_document_text: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/run_ocr/<document_id>')
@login_required
def run_ocr(document_id):
    """Run OCR on a document using EasyOCR for better results"""
    try:
        # Get document info
        if document_id not in document_processing_status:
            return jsonify({"success": False, "error": "Document not found"})
        
        doc = document_processing_status[document_id]
        pdf_filepath = doc.get('filepath')
        
        if not pdf_filepath or not os.path.exists(pdf_filepath):
            return jsonify({"success": False, "error": "PDF file not found"})
        
        # Run the advanced OCR directly
        text = extract_text_with_easyocr(pdf_filepath)
        
        if not text or text.strip() == "":
            return jsonify({"success": False, "error": "OCR could not extract any text from this document"})
            
        return jsonify({"success": True, "text": text})
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return jsonify({"success": False, "error": str(e)})

def extract_sentences(text):
    """
    Extract individual sentences from document text for analysis.
    This ensures we can highlight actual text that exists in the document.
    """
    import re
    
    # Clean the text - remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences (using common sentence-ending punctuation)
    # This regex handles common sentence endings while trying to avoid splitting
    # on periods in abbreviations, numbers, etc.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Further split long sentences into manageable chunks (for better highlighting)
    result = []
    for sentence in sentences:
        if len(sentence) > 200:  # If sentence is very long
            # Split on common delimiters within a sentence
            chunks = re.split(r'(?<=;|:|\n)\s+', sentence)
            result.extend(chunks)
        else:
            result.append(sentence)
    
    # Remove very short or empty sentences
    result = [s.strip() for s in result if len(s.strip()) > 10]
    
    return result

# Enhance monetary value extraction - add after preprocess_text function
def extract_monetary_values(text):
    """Extract monetary values from text using enhanced regex patterns for non-uniform formatting."""
    # Enhanced patterns for better monetary value detection
    money_patterns = [
        # Standard currency formats
        r'\$\s*[\d,]+\.?\d*(?:\s*(?:thousand|k|million|m|billion|b))?',              # $10,000, $10k, $1.5M
        r'USD\s*[\d,]+\.?\d*',                                                       # USD 10,000
        r'[\d,]+\.?\d*\s*(?:dollars?|USD|usd)',                                     # 10,000 dollars
        
        # European currencies
        r'[\d,]+\.?\d*\s*(?:euros?|EUR|eur|€)',                                     # 1,500 euros
        r'€\s*[\d,]+\.?\d*',                                                         # €1,500
        r'£\s*[\d,]+\.?\d*',                                                         # £1,500
        r'[\d,]+\.?\d*\s*(?:pounds?|GBP|gbp)',                                      # 1,500 pounds
        
        # Percentage values
        r'[\d,]+\.?\d*\s*%(?:\s*(?:per|of|annual|monthly|daily)(?:\s+\w+)*)?',     # 5% per annum
        
        # Range formats
        r'\$?\s*[\d,]+\.?\d*\s*(?:to|-|through)\s*\$?\s*[\d,]+\.?\d*',             # $500-$2,000 or 500 to 2,000
        
        # Fee and penalty formats
        r'(?:fee|penalty|fine|charge|cost|amount|sum|payment)\s*(?:of|:)?\s*\$?\s*[\d,]+\.?\d*', # fee of $500
        
        # Per unit formats
        r'\$?\s*[\d,]+\.?\d*\s*(?:per|/)\s*(?:day|month|year|hour|unit|item)',     # $100/day
        
        # Currency with decimals and separators (handles European formats too)
        r'[\d]+[,.][\d]{3}[,.][\d]{2}',                                             # 1,000.00 or 1.000,00
        
        # Late fees and interest rates
        r'(?:interest|rate|apr|apy)\s*(?:of|:)?\s*[\d,]+\.?\d*\s*%',               # interest of 5%
        
        # Minimum/maximum amounts
        r'(?:minimum|maximum|max|min|not\s+(?:less|more)\s+than)\s+\$?\s*[\d,]+\.?\d*', # minimum $1,000
        
        # Fractional amounts
        r'[\d]+\s*/\s*[\d]+\s*(?:of|percent|%)',                                   # 1/4 of, 3/4 percent
        
        # Written out currencies with numbers
        r'(?:cost|price|worth|value)\s+(?:is|of)?\s*[\d,]+\.?\d*\s*(?:dollars?|euros?|pounds?)', # cost is 1,500 euros
    ]
    
    extracted_values = []
    text_lower = text.lower()
    
    for pattern in money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            clean_match = match.strip()
            # Ensure we have meaningful values (not just currency symbols)
            if re.search(r'\d', clean_match):
                extracted_values.append(clean_match)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for value in extracted_values:
        normalized = re.sub(r'\s+', ' ', value.strip().lower())
        if normalized not in seen and len(normalized) > 1:
            seen.add(normalized)
            unique_values.append(value)
    
    return unique_values[:3]  # Return top 3 matches to avoid overwhelming results

# Add parallel processing for large documents - add after handle_large_document
def process_document_parallel(document_id, text, status_logs):
    """Process document in parallel for faster analysis."""
    send_status_update(document_id, "processing", "Preparing document for parallel processing", 10)
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Get document chunks
    chunks = truncate_for_model(processed_text)
    
    # If chunks is a string (not actually chunked), convert to list
    if isinstance(chunks, str):
        chunks = [chunks]
        
    send_status_update(document_id, "processing", f"Document split into {len(chunks)} segments for parallel processing", 20)
    
    # Create a partial function with fixed parameters
    chunk_analyzer = partial(analyze_chunk, document_id=document_id)
    
    # Process chunks in parallel with ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {executor.submit(chunk_analyzer, chunk, i): i for i, chunk in enumerate(chunks)}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                chunk_result = future.result()
                results.append(chunk_result)
                
                # Update progress
                completed += 1
                progress = 20 + (70 * completed / len(chunks))
                send_status_update(document_id, "processing", 
                                  f"Processed segment {completed}/{len(chunks)}", 
                                  progress)
                
            except Exception as e:
                status_logs.append(f"Error processing chunk {chunk_index}: {str(e)}")
                send_status_update(document_id, "processing", 
                                  f"Error in segment {chunk_index}", 
                                  progress)
    
    # Combine and deduplicate results
    combined_results = combine_results(results)
    
    return combined_results

# Helper function for parallel processing
def analyze_chunk(chunk, chunk_index, document_id):
    """Analyze a single document chunk."""
    try:
        # Extract sentences for this chunk
        chunk_sentences = extract_sentences(chunk)
        
        # Format sentences for analysis
        sentences_text = ""
        for i, sent in enumerate(chunk_sentences):
            sentences_text += f"[{i+1}] {sent}\n\n"
        
        # Analyze the chunk
        chunk_analysis = analyze_contract(sentences_text)
        
        # Enhance monetary values in results
        if 'clauses' in chunk_analysis:
            for clause in chunk_analysis['clauses']:
                # If amount not specified but exists in original text, try to extract
                if clause.get('amount') in ['N/A', 'Not specified', '...'] and clause.get('original_text'):
                    extracted_amounts = extract_monetary_values(clause['original_text'])
                    if extracted_amounts:
                        clause['amount'] = extracted_amounts[0]  # Use first found amount
        
        return chunk_analysis
    except Exception as e:
        return {"error": str(e), "chunk_index": chunk_index}

# Helper function to combine results from parallel processing
def combine_results(chunk_results):
    """Combine and deduplicate results from multiple chunks."""
    if not chunk_results:
        return {"clauses": []}
    
    all_clauses = []
    seen_clauses = set()
    
    # First, collect all clauses
    for result in chunk_results:
        if isinstance(result, dict) and 'clauses' in result:
            all_clauses.extend(result['clauses'])
    
    # Then deduplicate
    unique_clauses = []
    for clause in all_clauses:
        # Create a signature based on type and text
        clause_type = clause.get('type', '')
        exact_text = clause.get('exact_text', '')
        signature = f"{clause_type}:{exact_text}"
        
        if signature not in seen_clauses:
            seen_clauses.add(signature)
            unique_clauses.append(clause)
    
    # Renumber IDs
    for i, clause in enumerate(unique_clauses):
        clause['id'] = i + 1
    
    return {"clauses": unique_clauses}

# Modify handle_large_document to use parallel processing
def handle_large_document(text, status_logs):
    """Handle processing of large documents by splitting them into manageable chunks."""
    # Preserve the original function and add parallel processing capability
    document_id = None
    for log in status_logs:
        if isinstance(log, dict) and 'document_id' in log:
            document_id = log['document_id']
            break
    
    try:
        # Use parallel processing if we have a document_id
        if document_id:
            return process_document_parallel(document_id, text, status_logs)
        
        # Otherwise, use the original implementation
        status_logs.append("Using sequential processing (no document ID found)")
        
        # Split the text into chunks that can be processed independently
        text_length = len(text)
        chunk_size = text_length // 5  # Divide into 5 parts
        
        chunks = []
        for i in range(0, text_length, chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        # Process each chunk
        all_sentences = []
        for i, chunk in enumerate(chunks):
            chunk_sentences = extract_sentences(chunk)
            all_sentences.extend(chunk_sentences)
        
        # Prepare sentences for analysis
        sentences_text = ""
        for i, sent in enumerate(all_sentences):
            sentences_text += f"[{i+1}] {sent}\n\n"
        
        # Analyze the text
        result = analyze_contract(sentences_text)
        return result
        
    except Exception as e:
        status_logs.append(f"Error in document processing: {str(e)}")
        # Fall back to direct analysis if all else fails
        return analyze_contract(text)

# Add these imports if they don't exist already at the top of the file
import asyncio
import aiohttp
import json
from functools import partial
import time

# Add this function after extract_monetary_values function
async def analyze_chunk_async(session, chunk, chunk_index, document_id, overlap_context=""):
    """
    Asynchronously analyze a single document chunk using API calls.
    
    Args:
        session: aiohttp ClientSession for making HTTP requests
        chunk: Text chunk to analyze
        chunk_index: Index of this chunk in the document
        document_id: ID of the document being processed
        overlap_context: Text from adjacent chunks for context preservation
    
    Returns:
        Analysis result for this chunk
    """
    try:
        # Add overlap context if available
        if overlap_context:
            enhanced_chunk = f"{overlap_context}\n\n{chunk}"
        else:
            enhanced_chunk = chunk
            
        # Extract sentences for this chunk
        chunk_sentences = extract_sentences(enhanced_chunk)
        
        # Format sentences for analysis
        sentences_text = ""
        for i, sent in enumerate(chunk_sentences):
            sentences_text += f"[{i+1}] {sent}\n\n"
        
        # Prepare the prompt for contract analysis
        prompt = f"""
        You are analyzing a contract to identify CRITICAL BUSINESS RISKS that require management attention.
        Think like a business advisor, not a lawyer. Focus on risks that could:
        
        1. COST MONEY: Financial penalties, unexpected costs, payment obligations
        2. DISRUPT OPERATIONS: Service interruptions, compliance burdens, resource constraints
        3. CREATE LIABILITY: Legal exposure, unlimited damages, indemnification
        4. RESTRICT GROWTH: Non-compete clauses, exclusivity restrictions, IP limitations
        5. THREATEN CONTINUITY: Termination risks, key person dependencies, renewal uncertainties
        
        ANALYZE THESE CONTRACT SENTENCES AND CONSOLIDATE RELATED RISKS:
        
        {sentences_text}
        
        INSTRUCTIONS:
        - Combine similar risks into single, comprehensive analyses
        - Use business-friendly language (avoid legal jargon)
        - Quantify financial exposure wherever possible
        - Provide specific mitigation recommendations
        - Focus on TOP 5-8 MOST MATERIAL RISKS only
        - Each risk should answer: "What could go wrong and what would it cost us?"
        
        EXAMPLES OF GOOD BUSINESS TITLES:
        ❌ Bad: "Indemnification Clause"
        ✅ Good: "Unlimited Legal Liability for Third-Party Claims"
        
        ❌ Bad: "Material Adverse Change"  
        ✅ Good: "Lender Can Demand Immediate Repayment During Market Downturns"
        
        ❌ Bad: "Termination for Convenience"
        ✅ Good: "Client Can Cancel Anytime, Leaving Us with Unrecovered Costs"
        
        Return comprehensive risk analysis focusing on business impact and actionability.
        """
        
        # Make API call to analyze this chunk
        # This example assumes you're using an API endpoint, replace with your actual API logic
        api_response = await analyze_contract_api(session, prompt, document_id, chunk_index)
        
        # If you're using your existing analyze_contract function:
        # api_response = await loop.run_in_executor(None, analyze_contract, sentences_text)
        
        # Enhance monetary values in results
        if 'clauses' in api_response:
            for clause in api_response['clauses']:
                # If amount not specified but exists in original text, try to extract
                if clause.get('amount') in ['N/A', 'Not specified', '...', ''] and clause.get('original_text'):
                    extracted_amounts = extract_monetary_values(clause['original_text'])
                    if extracted_amounts:
                        clause['amount'] = extracted_amounts[0]  # Use first found amount
        
        return api_response
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return {"error": str(e), "chunk_index": chunk_index}

# Replace the analyze_contract_api function with this implementation that uses your existing analyze_contract function
async def analyze_contract_api(session, prompt, document_id, chunk_index):
    """Make async API call to language model for contract analysis"""
    try:
        # Create a reference to your existing analyze_contract function
        loop = asyncio.get_running_loop()
        
        # Run your existing analyze_contract function in a thread pool executor
        # This allows the CPU-bound analyze_contract to run without blocking the event loop
        response = await loop.run_in_executor(None, analyze_contract, prompt)
        
        # If analyze_contract already returns a dict, return it directly
        if isinstance(response, dict) and 'clauses' in response:
            return response
            
        # If analyze_contract returns something else (like JSON text), process it
        if isinstance(response, str):
            return extract_json_from_response(response)
            
        # Handle other return types
        return {"error": "Unexpected response format", "chunk_index": chunk_index}
    except Exception as e:
        print(f"API error for chunk {chunk_index}: {str(e)}")
        return {"error": str(e), "chunk_index": chunk_index}

def handle_large_document(text, status_logs):
    """
    Args:
        text: The document text
        status_logs: List for status logging

    Returns:
        Analysis results
    """
    # Get document ID from status logs if available
    document_id = None
    for log in status_logs:
        if isinstance(log, dict) and 'document_id' in log:
            document_id = log['document_id']
            break

    if not document_id:
        status_logs.append("Error: No document ID found in status logs")
        # Fallback: Create a basic summary first, then analyze
        try:
            # Try to create a summary from the text first
            basic_summary = deepseek_summarize_full(text[:50000])  # Use first 50k chars for summary
            return analyze_contract(basic_summary, text)  # Correct parameters: summary, original_text
        except Exception as e:
            status_logs.append(f"Fallback summary failed: {str(e)}")
            # Last resort: return mock results
            return {"clauses": []}

    try:
        # Create and run event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use the extractive summarization approach instead of async processing
            status_logs.append("Processing large document with extractive summarization")
            
            # Calculate target sentences based on document size
            estimated_sentences = len(text) // 80  # Rough estimate: 80 chars per sentence
            target_sentences = min(50, max(20, estimated_sentences // 50))  # 20-50 sentences
            
            status_logs.append(f"Creating extractive summary with {target_sentences} key sentences")
            extractive_summary = extractive_summarize_large_document(text, target_sentences)
            
            # Create AI summary from the extractive summary
            status_logs.append("Generating final summary from key sentences")
            summary = deepseek_summarize_full(extractive_summary)
            
            # Analyze the contract using the summaries
            status_logs.append("Analyzing contract for risks")
            clauses = analyze_contract(summary, extractive_summary)
            
            status_logs.append("Large document processing complete")
            return {"clauses": clauses}
        finally:
            loop.close()
    except Exception as e:
        error_msg = f"Error in parallel document processing: {str(e)}"
        status_logs.append(error_msg)
        print(error_msg)
        # Fall back to direct analysis
        return analyze_contract(text)

# Modify the process_document function to use handle_large_document for all documents
def process_document(filepath, document_id):
    """Process document in background with status updates"""
    try:
        # Update status to extracting
        document_processing_status[document_id]['status'] = 'extracting'
        document_processing_status[document_id]['current_step'] = 'Extracting text from PDF...'
        document_processing_status[document_id]['progress'] = 10
        
        # Extract text
        text = extract_text_robust(filepath)
        
        # Preprocess the text for better analysis
        clean_text = preprocess_text(text)
        
        # Update status to summarizing
        document_processing_status[document_id]['status'] = 'summarizing'
        document_processing_status[document_id]['current_step'] = 'Generating contract summary...'
        document_processing_status[document_id]['progress'] = 40
        
        # Generate summary
        status_logs = []  # Initialize status logs
        # Add document_id to status_logs so handle_large_document can find it
        status_logs.append({"document_id": document_id, "message": "Starting large document processing"})
        
        if len(clean_text) > 240000:
            try:
                # Use the extractive summarization approach for large documents
                status_logs.append("Processing large document with extractive summarization")
                
                # Calculate target sentences based on document size
                estimated_sentences = len(text) // 80  # Rough estimate: 80 chars per sentence
                target_sentences = min(50, max(20, estimated_sentences // 50))  # 20-50 sentences
                
                status_logs.append(f"Creating extractive summary with {target_sentences} key sentences")
                extractive_summary = extractive_summarize_large_document(text, target_sentences)
                
                # Create AI summary from the extractive summary
                status_logs.append("Generating final summary from key sentences")
                summary = deepseek_summarize_full(extractive_summary)
                
                # Analyze the contract using the summaries
                status_logs.append("Analyzing contract for risks")
                clauses = analyze_contract(summary, extractive_summary)
                
                status_logs.append("Large document processing complete")
                return {"clauses": clauses}
                
            except Exception as e:
                error_msg = f"Error in large document processing: {str(e)}"
                status_logs.append(error_msg)
                print(error_msg)
                # Fall back to basic analysis with truncated text
                try:
                    truncated_text = text[:50000] + "... [Document truncated for processing]"
                    summary = deepseek_summarize_full(truncated_text)
                    clauses = analyze_contract(summary)
                    return {"clauses": clauses}
                except Exception as e2:
                    status_logs.append(f"Fallback analysis also failed: {str(e2)}")
                    return {"clauses": []}
        else:
            summary = deepseek_summarize_full(clean_text)
        
        # Update status to analyzing
        document_processing_status[document_id]['status'] = 'analyzing'
        document_processing_status[document_id]['current_step'] = 'Identifying risk factors...'
        document_processing_status[document_id]['progress'] = 70
        
        # Extract sentences for better highlighting
        sentences = extract_sentences(clean_text)
        status_logs.append(f"Extracted {len(sentences)} sentences for risk analysis")
        
        # Analyze contract for risks, using appropriate text size
        try:
            if len(clean_text) > 240000:
                # For large documents, use extractive summary for risk analysis too
                status_logs.append("Using extractive summary for risk analysis")
                extractive_summary_for_analysis = extractive_summarize_large_document(clean_text, target_sentences=30)
                clauses = analyze_contract(summary, extractive_summary_for_analysis)
            else:
                # For normal documents, use full text
                clauses = analyze_contract(summary, clean_text)
        except Exception as e:
            status_logs.append(f"Risk analysis failed: {str(e)}")
            # Fallback to basic analysis without original text
            try:
                clauses = analyze_contract(summary)
            except Exception as e2:
                status_logs.append(f"Basic analysis also failed: {str(e2)}")
                clauses = []  # Empty results as last resort
        
        # Sort clauses by risk level
        clauses = sorted(clauses, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x["risk"], 3))
        
        # Enhanced clause matching - store the exact text from the document for highlighting
        for clause in clauses:
            # First check if we have exact_text from the AI model
            if "exact_text" in clause and clause["exact_text"]:
                highlight_text = clause["exact_text"]
            # Check if we already have original_text from sentence-based analysis
            elif "original_text" in clause and clause["original_text"]:
                highlight_text = clause["original_text"]
                # Use the clause description as fallback
            else:
                highlight_text = clause.get("clause", "")
            
            # Advanced text matching using enhanced algorithm for layout-preserved text
            # 1. First try exact match with enhanced algorithm
            start_offset = enhanced_text_matching(clean_text, highlight_text)
            
            # 2. If enhanced matching fails, try traditional approaches
            if start_offset == -1:
                # Try exact traditional match
                start_offset = clean_text.find(highlight_text)
            
            # 3. If exact match fails, try case-insensitive match
            if start_offset == -1:
                lower_text = clean_text.lower()
                lower_highlight = highlight_text.lower()
                start_offset = lower_text.find(lower_highlight)
                if start_offset != -1:
                    # If found, use the actual text at this position (preserving case)
                    highlight_text = clean_text[start_offset:start_offset + len(highlight_text)]
            
            # 4. If that fails, try with smaller segments - first check type field
            if start_offset == -1 and "type" in clause:
                clause_type = clause.get("type")
                if clause_type and len(clause_type) > 3:  # Only use meaningful types
                    type_offset = clean_text.lower().find(clause_type.lower())
                    if type_offset != -1:
                        # Found the clause type, get surrounding context
                        context_start = max(0, type_offset - 20)
                        context_end = min(len(clean_text), type_offset + len(clause_type) + 100)
                        highlight_text = clean_text[context_start:context_end]
                        start_offset = context_start
            
            # 5. Try with key words from the clause text if still no match
            if start_offset == -1:
                # Get significant words from highlight text
                import re
                words = re.findall(r'\b\w{4,}\b', highlight_text.lower())
                if words:
                    # Look for clusters of significant words
                    matches = []
                    for word in words:
                        word_offset = clean_text.lower().find(word)
                        if word_offset != -1:
                            matches.append((word_offset, word))
                    
                    if matches:
                        # Sort matches by position
                        matches.sort(key=lambda x: x[0])
                        # Find clusters (words close to each other)
                        if len(matches) > 1:
                            for i in range(len(matches) - 1):
                                current_pos = matches[i][0]
                                next_pos = matches[i+1][0]
                                # If words are close (within 100 chars)
                                if next_pos - current_pos < 100:
                                    context_start = max(0, current_pos - 10)
                                    context_end = min(len(clean_text), next_pos + len(matches[i+1][1]) + 10)
                                    highlight_text = clean_text[context_start:context_end]
                                    start_offset = context_start
                                    break
                        # If no clusters or just one word, use first significant word with context
                        if start_offset == -1 and matches:
                            first_match = matches[0]
                            context_start = max(0, first_match[0] - 10)
                            context_end = min(len(clean_text), first_match[0] + len(first_match[1]) + 50)
                            highlight_text = clean_text[context_start:context_end]
                            start_offset = context_start
            
            # If any matching approach succeeded
            if start_offset != -1:
                end_offset = start_offset + len(highlight_text)
                # Add offsets to clause data
                clause["textOffsets"] = {
                    "start": start_offset,
                    "end": end_offset,
                }
                
                # Also store a context snippet for highlighting
                context_start = max(0, start_offset - 50)
                context_end = min(len(clean_text), end_offset + 50)
                clause["contextSnippet"] = clean_text[context_start:context_end]
                
                # Store the highlight text separately
                clause["highlight_text"] = highlight_text
            else:
                # If all matching approaches failed, log it
                print(f"Warning: Could not find a match for clause: {clause.get('type', 'Unknown type')}")
                # Still store a default context to ensure something appears
                if "type" in clause:
                    clause["contextSnippet"] = f"{clause.get('type')}: {clause.get('clause', '')}"
                    clause["highlight_text"] = clause.get('clause', '')
        
        # Preview text for display
        preview_text = clean_text[:1000] + "..." if len(clean_text) > 1000 else clean_text
        
        # Update status to complete
        document_processing_status[document_id]['status'] = 'complete'
        document_processing_status[document_id]['current_step'] = 'Analysis complete!'
        document_processing_status[document_id]['progress'] = 100
        document_processing_status[document_id]['complete'] = True
        document_processing_status[document_id]['preview_text'] = preview_text
        document_processing_status[document_id]['summary'] = summary
        document_processing_status[document_id]['clauses'] = clauses
        document_processing_status[document_id]['processing_time'] = int(time.time() - document_processing_status[document_id]['start_time'])
        document_processing_status[document_id]['status_logs'] = status_logs
        document_processing_status[document_id]['pdf_path'] = filepath
        document_processing_status[document_id]['original_text'] = clean_text  # Ensure we store the complete text for later use
        
        # Store results (you'd implement caching here)
        print(f"Processing complete: {filepath}")
    except Exception as e:
        # Update status to error
        document_processing_status[document_id]['status'] = 'error'
        document_processing_status[document_id]['current_step'] = f'Error: {str(e)}'
        document_processing_status[document_id]['complete'] = True
        document_processing_status[document_id]['error'] = str(e)
        print(f"Error processing document: {e}")

def extract_text_with_openai_vision(filepath):
    """
    Extract text from PDF using OpenAI GPT-4.1 Mini Vision API - the latest and most accurate method.
    GPT-4.1 Mini offers significant improvements: 83% cost reduction, nearly half the latency,
    while matching or exceeding GPT-4o performance in many benchmarks.
    """
    # Check if OpenAI client is available
    if not openai_client:
        print("❌ OpenAI API key not configured - skipping AI extraction")
        return ""
        
    try:
        print(f"🤖 Starting GPT-4o vision extraction for: {filepath}")
        
        # Convert PDF to high-quality images
        images = convert_from_path(filepath, dpi=300, fmt='JPEG')  # High DPI for better quality
        extracted_text = []
        
        # Process pages (GPT-4.1 Mini is even more efficient and cost-effective for large documents)
        max_pages = 25 if OPENAI_VISION_MODEL == "gpt-4.1-mini" else 15 if OPENAI_VISION_MODEL == "gpt-4o" else 10  # GPT-4.1 Mini can handle more pages efficiently
        
        for page_num, img in enumerate(images[:max_pages]):
            try:
                print(f"📄 Processing page {page_num + 1}/{min(len(images), max_pages)} with {OPENAI_VISION_MODEL}...")
                
                # Convert PIL image to base64
                import io
                img_buffer = io.BytesIO()
                # Optimize image quality for best OCR results
                img.save(img_buffer, format='JPEG', quality=95, optimize=True)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Enhanced prompt for contract text extraction
                system_prompt = """You are an expert document transcription AI specializing in legal and business contracts. Your goal is to extract text with PERFECT ACCURACY for contract analysis.

🎯 CRITICAL REQUIREMENTS:

1. **EXTRACT ALL TEXT** - Include every word, number, symbol, and punctuation mark visible in the document
2. **PRESERVE EXACT FORMATTING** - Maintain original spacing, line breaks, indentation, and paragraph structure
3. **MAINTAIN LEGAL PRECISION** - Contract text must be 100% accurate for legal analysis
4. **CAPTURE ALL DETAILS** - Include headers, footers, page numbers, signatures, dates, amounts, percentages

📋 SPECIAL FOCUS AREAS:
- Financial terms (dollar amounts, percentages, payment schedules)
- Legal obligations and penalties
- Dates and deadlines (contract periods, notice requirements)
- Party names and contact information
- Signature blocks and execution details
- Termination and cancellation clauses
- Liability and indemnification sections

🔍 TEXT EXTRACTION STANDARDS:
- Extract monetary amounts EXACTLY: $1,500.00, 5.25%, etc.
- Preserve all dates in original format
- Include reference numbers and case numbers exactly as shown
- Maintain original capitalization and punctuation
- Keep table structures and alignments where possible

⚠️ QUALITY REQUIREMENTS:
- This text will be used for AI risk analysis
- Any missing or incorrect text could cause missed legal risks
- Prioritize completeness over formatting perfection
- If text is unclear, include your best interpretation with [UNCLEAR: text] notation

OUTPUT: Return ONLY the extracted text with preserved structure. No explanations or commentary."""

                user_prompt = f"""Extract ALL text from this contract document page using {OPENAI_VISION_MODEL}. Focus on accuracy and completeness - every word matters for legal analysis.

Include:
- All contract terms and conditions
- Financial amounts and percentages  
- Dates and deadlines
- Party names and signatures
- Legal clauses and obligations
- Headers, footers, and page elements

Maintain the original document structure and formatting as much as possible."""

                # Call OpenAI GPT-4.1 Mini Vision API with optimized settings
                response = openai_client.chat.completions.create(
                    model=OPENAI_VISION_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}",
                                        "detail": "high"  # Use high detail for maximum accuracy
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4096,  # GPT-4.1 Mini can generate up to 32K tokens but 4K is sufficient for most pages
                    temperature=0.0,  # Zero temperature for consistency
                    timeout=45.0  # GPT-4.1 Mini is nearly 50% faster, so shorter timeout
                )
                
                page_text = response.choices[0].message.content.strip()
                if page_text:
                    extracted_text.append(page_text)
                    print(f"✅ Page {page_num + 1}: Extracted {len(page_text)} characters")
                else:
                    print(f"⚠️ Page {page_num + 1}: No text extracted")
                    
            except Exception as e:
                print(f"❌ Error processing page {page_num + 1}: {e}")
                continue
        
        # Combine all pages with clear separators
        if extracted_text:
            full_text = "\n\n=== PAGE BREAK ===\n\n".join(extracted_text)
            print(f"🎉 GPT-4.1 Mini extraction completed successfully!")
            print(f"📊 Total: {len(full_text)} characters from {len(extracted_text)} pages")
            
            # Quality check
            word_count = len(full_text.split())
            if word_count < 50:
                print(f"⚠️ Warning: Extracted text seems short ({word_count} words). Document may be image-based or corrupted.")
            
            return full_text
        else:
            print("❌ No text was extracted from any pages")
            return ""
        
    except Exception as e:
        print(f"❌ Error in GPT-4o text extraction: {e}")
        return ""

@app.route('/test_openai_connection')
@login_required
def test_openai_connection():
    """Test OpenAI API connection and configuration"""
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI client not initialized',
                'details': 'API key may not be configured'
            })
        
        if not OPENAI_API_KEY:
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not found',
                'details': 'OPENAI_API_KEY environment variable not set'
            })
        
        # Test basic API connection with a simple text completion
        print("🧪 Testing OpenAI API connection...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Respond with 'API connection successful' if you receive this."}
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'message': 'OpenAI API connection successful',
            'api_response': result,
            'model_used': "gpt-4o-mini",
            'vision_model_configured': OPENAI_VISION_MODEL,
            'client_initialized': True
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'OpenAI API connection failed'
        })

@app.route('/test_openai_vision/<document_id>')
@login_required
def test_openai_vision(document_id):
    """Test OpenAI vision text extraction capabilities"""
    if not is_valid_uuid(document_id):
        return jsonify({'error': 'Invalid document ID format'}), 400
        
    if document_id not in document_processing_status:
        return jsonify({'error': 'Document not found'}), 404
    
    # Security check: ensure user owns this document
    status = document_processing_status[document_id]
    if status.get('user_id') != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    filepath = status.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'PDF file not found'}), 404
    
    try:
        print("🧪 Testing OpenAI Vision text extraction...")
        
        # Test OpenAI Vision extraction
        start_time = time.time()
        extracted_text = extract_text_with_openai_vision(filepath)
        extraction_time = time.time() - start_time
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'OpenAI Vision extraction returned minimal text',
                'text_length': len(extracted_text) if extracted_text else 0,
                'extraction_time': extraction_time,
                'model_used': OPENAI_VISION_MODEL
            })
        
        # Basic analysis of the extracted text
        word_count = len(extracted_text.split())
        line_count = len(extracted_text.split('\n'))
        char_count = len(extracted_text)
        
        # Check for common contract elements
        contract_indicators = {
            'monetary_amounts': len(re.findall(r'\$[\d,]+(?:\.\d{2})?', extracted_text)),
            'dates': len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', extracted_text)),
            'percentages': len(re.findall(r'\d+(?:\.\d+)?%', extracted_text)),
            'signatures': len(re.findall(r'signature|signed|sign|signatory', extracted_text, re.IGNORECASE)),
            'legal_terms': len(re.findall(r'agreement|contract|terms|conditions|party|parties', extracted_text, re.IGNORECASE))
        }
        
        # Check if this looks like a contract
        is_contract = (
            contract_indicators['legal_terms'] > 2 or 
            contract_indicators['monetary_amounts'] > 0 or
            'agreement' in extracted_text.lower() or
            'contract' in extracted_text.lower()
        )
        
        return jsonify({
            'success': True,
            'message': f'OpenAI {OPENAI_VISION_MODEL} extraction completed successfully',
            'stats': {
                'character_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'extraction_time_seconds': round(extraction_time, 2),
                'model_used': OPENAI_VISION_MODEL
            },
            'contract_analysis': contract_indicators,
            'document_type': 'Contract/Legal Document' if is_contract else 'General Document',
            'text_preview': extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
            'quality_assessment': {
                'text_length': 'Good' if word_count > 200 else 'Fair' if word_count > 50 else 'Poor',
                'contract_indicators': 'High' if sum(contract_indicators.values()) > 5 else 'Medium' if sum(contract_indicators.values()) > 2 else 'Low'
            }
        })
        
    except Exception as e:
        print(f"❌ Error in OpenAI Vision test: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'OpenAI Vision extraction failed',
            'model_used': OPENAI_VISION_MODEL
        })

# Flask app runner
if __name__ == '__main__':
    print("🚀 Starting Contract Analyzer Flask Application...")
    print(f"📊 Using OpenAI Vision Model: {OPENAI_VISION_MODEL}")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Not configured'}")
    print(f"🤖 DeepSeek API Key: {'✅ Configured' if DEESEEK_API_KEY else '❌ Not configured'}")
    print("🌐 Starting server...")
    
    if OPENAI_VISION_MODEL == "gpt-4.1-mini":
        print("🌟 Using latest GPT-4.1 Mini: 83% cost reduction, 50% faster, superior performance!")
    
    # Run the Flask application
    app.run(
        host='127.0.0.1',
        port=5001,  # Changed from 5000 to avoid AirPlay conflicts
        debug=True,  # Enable debug mode for development
        threaded=True  # Enable threading for concurrent requests
    )

def enhanced_text_matching(document_text, search_text):
    """
    Enhanced text matching algorithm for better clause highlighting.
    Handles variations in spacing, formatting, and OCR artifacts.
    
    Args:
        document_text: The full document text to search in
        search_text: The text to find within the document
    
    Returns:
        int: Start position if found, -1 if not found
    """
    if not document_text or not search_text:
        return -1
    
    # Clean and normalize both texts for comparison
    def normalize_text(text):
        import re
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common OCR artifacts
        text = text.replace('|', 'l').replace('0', 'O')
        return text.lower()
    
    # First try exact match (fastest)
    exact_match = document_text.find(search_text)
    if exact_match != -1:
        return exact_match
    
    # Try case-insensitive match
    lower_doc = document_text.lower()
    lower_search = search_text.lower()
    case_match = lower_doc.find(lower_search)
    if case_match != -1:
        return case_match
    
    # Try normalized text matching (handles spacing variations)
    normalized_doc = normalize_text(document_text)
    normalized_search = normalize_text(search_text)
    
    norm_match = normalized_doc.find(normalized_search)
    if norm_match != -1:
        # Find the corresponding position in the original text
        return find_original_position(document_text, norm_match, len(normalized_search))
    
    # Try fuzzy matching with key phrases (last resort)
    return fuzzy_phrase_match(document_text, search_text)

def find_original_position(original_text, normalized_position, normalized_length):
    """
    Map a position from normalized text back to original text.
    This is a simplified version - in practice, you'd need more sophisticated mapping.
    """
    # For simplicity, return the normalized position
    # In a production system, you'd maintain a mapping between normalized and original positions
    return min(normalized_position, len(original_text) - 1) if normalized_position < len(original_text) else -1

def fuzzy_phrase_match(document_text, search_text):
    """
    Find text using fuzzy matching of key phrases.
    Extracts significant words and looks for clusters of matches.
    """
    import re
    
    # Extract significant words (4+ characters, not common words)
    stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
    
    search_words = [w.lower() for w in re.findall(r'\b\w{4,}\b', search_text) if w.lower() not in stop_words]
    
    if not search_words:
        return -1
    
    # Find positions of each significant word
    word_positions = []
    doc_lower = document_text.lower()
    
    for word in search_words:
        pos = doc_lower.find(word)
        if pos != -1:
            word_positions.append(pos)
    
    if not word_positions:
        return -1
    
    # Return the position of the first found word
    return min(word_positions)

#!/usr/bin/env python3
"""
Enhanced Contract Analyzer Setup Script
This script helps set up the improved contract analysis application with all necessary dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"✓ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"  Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {description} failed")
        print(f"  Command: {command}")
        print(f"  Error output: {e.stderr}")
        return False

def check_system_requirements():
    """Check if required system packages are available."""
    print("🔍 Checking system requirements...")
    
    # Check for Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check for Tesseract (required for pytesseract)
    if not run_command("tesseract --version", "Checking Tesseract OCR"):
        print("⚠️  Tesseract not found. Please install it:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu: sudo apt-get install tesseract-ocr")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    # Check for poppler (required for pdf2image)
    if not run_command("pdftoppm -h", "Checking Poppler utils"):
        print("⚠️  Poppler not found. Please install it:")
        print("   macOS: brew install poppler")
        print("   Ubuntu: sudo apt-get install poppler-utils")
        print("   Windows: Download from https://blog.alivate.com.au/poppler-windows/")
        return False
    
    return True

def install_python_dependencies():
    """Install Python package dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def setup_environment():
    """Set up environment variables and configuration."""
    print("⚙️  Setting up environment...")
    
    # Create .env file template if it doesn't exist
    env_file = ".env"
    if not os.path.exists(env_file):
        env_template = """# Contract Analyzer Environment Configuration
# Copy this file to .env and fill in your API keys

# DeepSeek API Configuration (Required)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com

# Optional: OpenAI API Key (for fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
OFFLINE_MODE=False
FLASK_ENV=development
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        print(f"✓ Created {env_file} template")
        print("  Please edit .env file and add your API keys")
    else:
        print(f"✓ {env_file} already exists")
    
    # Create uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"✓ Created {uploads_dir} directory")
    else:
        print(f"✓ {uploads_dir} directory exists")
    
    return True

def verify_installation():
    """Verify that key components are working."""
    print("🧪 Verifying installation...")
    
    try:
        # Test imports
        import flask
        import easyocr
        import cv2
        import fitz
        import pytesseract
        print("✓ All Python packages imported successfully")
        
        # Test OCR functionality
        reader = easyocr.Reader(['en'], gpu=False)
        print("✓ EasyOCR initialized successfully")
        
        print("✅ Installation verification complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Enhanced Contract Analyzer Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please install missing dependencies.")
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Failed to install Python dependencies.")
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        print("\n❌ Failed to set up environment.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file and add your DeepSeek API key")
    print("2. Run the application: python app.py")
    print("3. Open your browser to http://localhost:5001")
    print("\nFeatures enhanced:")
    print("• Improved monetary value extraction")
    print("• Better OCR for non-uniform documents")
    print("• Smart clause filtering (reduces noise)")
    print("• Enhanced document processing")
    print("• Parallel processing capabilities")

if __name__ == "__main__":
    main() 
# 🤖 AI Contract Analyzer

**Production-Ready Contract Risk Analysis using GPT-4.1 Mini & DeepSeek v3**

Transform your contract review process with AI-powered risk detection, intelligent highlighting, and comprehensive business impact analysis.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--Mini-orange.svg)
![DeepSeek](https://img.shields.io/badge/DeepSeek-v3-purple.svg)

## ✨ Features

### 🎯 **Core Functionality**
- **AI-Powered Text Extraction**: GPT-4.1 Mini vision for 95%+ accuracy
- **Advanced Risk Analysis**: DeepSeek v3 for comprehensive contract evaluation
- **Smart Risk Scoring**: 0-100 scale using 5-criteria methodology
- **Real-time PDF Highlighting**: Visual clause identification in original documents
- **Professional UI**: Chrome-style tabs with modern responsive design

### 🔧 **Technical Capabilities**
- **Multi-Engine Text Extraction**: GPT-4.1 Mini → PyMuPDF → EasyOCR → Tesseract fallback
- **Background Processing**: Async document analysis with real-time status updates
- **Enhanced Text Matching**: Advanced algorithms for precise clause highlighting
- **User Authentication**: Secure session management and document ownership
- **Scalable Architecture**: Handles documents from 1-200+ pages

### 📊 **Risk Analysis**
- **Financial Impact Assessment**: Identifies costs, penalties, and monetary obligations
- **Business Disruption Evaluation**: Operational risk and compliance burden analysis
- **Legal Liability Detection**: Unlimited damages, indemnification, and exposure risks
- **Termination Risk Analysis**: Exit costs, cancellation terms, and continuity threats
- **Compliance Requirement Mapping**: Regulatory obligations and deadline tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- DeepSeek API Key
- Poppler (for PDF processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KevinChoi05/ai-contract-analyzer.git
   cd ai-contract-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Poppler (macOS)**
   ```bash
   brew install poppler
   ```

4. **Set up environment variables**
   Create a `.env` file or set environment variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   OPENAI_VISION_MODEL=gpt-4.1-mini
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5001`

## 🎯 Usage

1. **Register/Login**: Create an account or log in
2. **Upload Contract**: Drag & drop or select PDF files
3. **AI Analysis**: Watch real-time processing with status updates
4. **Review Results**: Examine risk scores, highlighted clauses, and recommendations
5. **Export/Share**: Save analysis results or share with team members

## 🏗️ Architecture

### AI Pipeline
```
PDF Upload → GPT-4.1 Mini Vision → Text Extraction → DeepSeek v3 Analysis → Risk Scoring → Highlighting
```

### Risk Scoring Methodology
- **Financial Impact (30%)**: Potential costs and monetary exposure
- **Business Disruption (25%)**: Operational impact and resource requirements  
- **Legal Risk (20%)**: Compliance obligations and legal exposure
- **Likelihood (15%)**: Probability of risk occurrence
- **Mitigation Difficulty (10%)**: Ease of risk resolution

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for GPT-4.1 Mini
- `DEEPSEEK_API_KEY`: DeepSeek API key for analysis
- `OPENAI_VISION_MODEL`: Model selection (gpt-4.1-mini, gpt-4o, gpt-4o-mini)
- `OFFLINE_MODE`: Demo mode without API calls

### Model Selection
- **gpt-4.1-mini**: 83% cost reduction, 50% faster, recommended
- **gpt-4o**: Higher accuracy for complex documents
- **gpt-4o-mini**: Budget option for basic analysis

## 🎮 Demo Mode

Set `OFFLINE_MODE=True` to run without API keys using mock data:
```bash
export OFFLINE_MODE=True
python app.py
```

## 📞 Support

For support, open an issue on GitHub.

---

**Made with ❤️ using AI** - Transforming contract review with cutting-edge technology 
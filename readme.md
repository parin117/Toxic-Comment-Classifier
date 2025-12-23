# 🛡️ Toxic Comment Classification & Stabilization System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Platform-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An end-to-end ML pipeline for detecting and stabilizing toxic comments using BERT, deployed on Google Cloud Platform**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 About

This project is a comprehensive **Toxic Comment Classification and Stabilization System** that automatically detects toxic comments in online platforms and provides AI-powered alternatives. The system leverages:

- **BERT-based NLP models** for multi-label toxicity classification
- **Kubeflow Pipelines** for automated ML workflows on Google Cloud
- **Google Cloud Run** for scalable API deployment
- **Google Gemini AI** for intelligent comment stabilization
- **Gradio** for an intuitive web-based user interface

The system classifies comments into six toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`, and uses AI to generate non-toxic alternatives when toxicity is detected.

---

## ✨ Features

- ✅ **Multi-label Classification**: Detects 6 types of toxicity in comments
- ✅ **BERT Fine-tuning**: State-of-the-art transformer-based model for text classification
- ✅ **Automated ML Pipelines**: Kubeflow Pipelines for reproducible ML workflows
- ✅ **Cloud Deployment**: Scalable API deployment on Google Cloud Run
- ✅ **AI-Powered Stabilization**: Google Gemini AI generates non-toxic comment alternatives
- ✅ **Interactive UI**: User-friendly Gradio interface for real-time inference
- ✅ **Dockerized**: Containerized application for easy deployment
- ✅ **RESTful API**: Flask-based API for easy integration
- ✅ **Cloud Storage Integration**: Seamless data management with Google Cloud Storage

---

## 🛠️ Tech Stack

### Machine Learning & NLP
- **PyTorch**: Deep learning framework for model training
- **Transformers (Hugging Face)**: BERT model and tokenizers
- **Pandas**: Data manipulation and preprocessing

### Cloud & Infrastructure
- **Google Cloud Platform (GCP)**: Cloud infrastructure
- **Google Cloud Storage (GCS)**: Data storage
- **Vertex AI**: ML model management and pipelines
- **Kubeflow Pipelines (KFP)**: ML workflow orchestration
- **Google Cloud Run**: Serverless API deployment
- **Docker**: Containerization

### Backend & API
- **Flask**: REST API framework
- **Python 3.8**: Programming language

### Frontend
- **Gradio**: Interactive web interface
- **LangChain**: Google Gemini AI integration

### Tools & Libraries
- **google-cloud-aiplatform**: GCP AI Platform SDK
- **google-cloud-storage**: GCS client library
- **google-cloud-pipeline-components**: Pre-built pipeline components

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│   Gradio UI     │─────▶│  Flask API       │
│  (Frontend)     │      │  (GCP Cloud Run) │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  BERT Model      │
                         │  (Classification)│
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
         ┌──────────────────┐      ┌──────────────────┐
         │  Toxic?          │      │  Gemini AI       │
         │  Yes/No          │      │  (Stabilization) │
         └──────────────────┘      └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Results Display │
         └──────────────────┘
```

**Training Pipeline Architecture:**
```
Data Collection → Preprocessing → Tokenization → BERT Training → Model Export → Cloud Storage
```

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Python 3.8+**
- **Docker** (for containerization)
- **Google Cloud SDK (gcloud CLI)** (for deployment)
- **Git**

### Required Accounts & Services
- **Google Cloud Platform Account** with billing enabled
- **Google Gemini API Key** (from [Google AI Studio](https://ai.google.dev/aistudio))
- **Kaggle Account** (for dataset access)

### Required GCP Services
The following APIs must be enabled in your GCP project:
- **IAM API**
- **Compute Engine API**
- **Vertex AI API**
- **Artifact Registry API**
- **Cloud Storage API**
- **Notebooks API**

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/parin117/Toxic-Comment-Classifier.git
cd Toxic-Comment-Classifier
```

### 2. Set Up Google Cloud Platform

#### 2.1 Create a GCP Project
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable billing for the project
4. Note down your **Project ID**

#### 2.2 Enable Required APIs
Enable the following APIs in your GCP project:
- IAM API
- Compute Engine API
- Vertex AI API
- Artifact Registry API
- Cloud Storage API
- Notebooks API

#### 2.3 Create a Cloud Storage Bucket
```bash
# Using gcloud CLI
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l asia-south1 gs://YOUR_BUCKET_NAME
```

#### 2.4 Create a Service Account
1. Go to **IAM & Admin > Service Accounts**
2. Create a new service account
3. Assign the following roles:
   - AI Platform Admin
   - Compute Admin
   - Owner
   - Storage Admin
   - Storage Object Admin
   - Vertex AI Service Admin
   - Artifact Registry Admin
4. Download the service account JSON key

#### 2.5 Authenticate gcloud
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth configure-docker
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Install Required Python Packages

```bash
pip install --upgrade google-cloud-aiplatform \
                      google-cloud-storage \
                      kfp \
                      google-cloud-pipeline-components \
                      torch \
                      transformers \
                      flask \
                      gradio \
                      langchain-google-genai \
                      python-dotenv \
                      pandas
```

### 5. Set Up Kaggle API (for Dataset)

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to **API** section
3. Click **Create New API Token**
4. Download `kaggle.json`
5. Place it in your project directory

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Cloud Configuration
PROJECT_ID=your-gcp-project-id
REGION=asia-south1
BUCKET_URI=gs://your-bucket-name
BUCKET_NAME=your-bucket-name
SERVICE_ACCOUNT=your-service-account@project-id.iam.gserviceaccount.com
PIPELINE_ROOT=gs://your-bucket-name/pipeline_root/toxic

# Google Gemini API
GOOGLE_API_KEY=your-gemini-api-key

# Flask API Configuration
FLASK_APP=detox.py
FLASK_RUN_HOST=0.0.0.0
PORT=8080
```

### Update Configuration in Code

Edit the configuration in `workflow.ipynb`:

```python
PROJECT_ID = "your-gcp-project-id"
REGION = "asia-south1"
BUCKET_URI = "gs://your-bucket-name"
BUCKET_NAME = "your-bucket-name"
SERVICE_ACCOUNT = "your-service-account@project-id.iam.gserviceaccount.com"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/toxic"
```

---

## 💻 Usage

### Local Development

#### 1. Running the Flask API Locally

```bash
# Navigate to the app directory
cd toxic_comment_app

# Run the Flask app
python detox.py
```

The API will be available at `http://localhost:8080`

#### 2. Testing the API

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment"}'
```

#### 3. Running the Gradio UI

```bash
# Activate virtual environment
source venv/bin/activate

# Run the Gradio app
python app.py
```

The UI will be available at a local URL and a public Gradio share URL.

### Using the Web Interface

1. **Enter a Comment**: Type or paste a comment in the input box
2. **Classify**: Click submit to analyze the comment
3. **View Results**: 
   - Toxicity categories are displayed
   - If toxic, a stabilized version is generated using Gemini AI
4. **Try Examples**: Use the provided example comments for testing

---

## 📁 Project Structure

```
ToxicCommentClassifier-main/
│
├── images/                          # Screenshots and documentation images
│   ├── frontend*.png               # UI screenshots
│   ├── img*.png                    # Setup guide images
│   └── kaggle*.png                 # Kaggle setup images
│
├── toxic_comment_app/               # Flask API and Gradio UI
│   ├── detox.py                    # Flask API for model inference
│   ├── app.py                      # Gradio UI application
│   ├── content/                    # Trained model files
│   │   ├── pytorch_model.bin
│   │   └── config.json
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Docker configuration
│   └── .env                        # Environment variables
│
├── workflow.ipynb                   # Complete Jupyter notebook with:
│                                   # - GCP setup guide
│                                   # - Data preprocessing pipeline
│                                   # - Model training pipeline
│                                   # - Deployment instructions
│
└── readme.md                        # This file
```

---

## 📚 API Documentation

### Endpoint: `/predict`

Classify a comment for toxicity.

**Request:**
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "text": "Your comment text here"
}
```

**Response:**
- **Status Code**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "toxic": 1,
  "severe_toxic": 0,
  "obscene": 0,
  "threat": 0,
  "insult": 1,
  "identity_hate": 0
}
```

**Error Response:**
- **Status Code**: `400 Bad Request` or `500 Internal Server Error`
- **Body**:
```json
{
  "error": "Error message description"
}
```

### Toxicity Categories

The model classifies comments into 6 categories:

1. **toxic**: General toxic language
2. **severe_toxic**: Highly toxic content
3. **obscene**: Obscene or vulgar language
4. **threat**: Threatening language
5. **insult**: Insulting language
6. **identity_hate**: Hate speech targeting identity

---

## 🚀 Deployment

### Deploying to Google Cloud Run

#### 1. Build Docker Image

```bash
cd toxic_comment_app
docker build -t gcr.io/YOUR_PROJECT_ID/toxic-comment-classifier .
```

#### 2. Push to Google Container Registry

```bash
docker push gcr.io/YOUR_PROJECT_ID/toxic-comment-classifier
```

#### 3. Deploy to Cloud Run

```bash
gcloud run deploy toxic-comment-api \
  --image gcr.io/YOUR_PROJECT_ID/toxic-comment-classifier \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --port 8080
```

#### 4. Get Deployment URL

After deployment, you'll receive a URL like:
```
https://toxic-comment-api-XXXXX.asia-south1.run.app
```

Update `SERVER_URL` in `app.py` with this URL.

### Running ML Pipelines on Vertex AI

Refer to `workflow.ipynb` for detailed instructions on:
- Setting up Kubeflow Pipelines
- Running data preprocessing pipeline
- Training the BERT model
- Exporting the trained model

---

## 📸 Screenshots

### Web Interface
![Frontend UI](images/frontend15.png)

### Classification Results
![Classification](images/frontend14.png)

### Deployment Dashboard
![GCP Dashboard](images/frontend8.png)

*Note: See the `images/` directory for more screenshots.*

---

## 📊 Dataset

This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle.

- **Source**: [Kaggle Competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Size**: ~160k training samples
- **Format**: CSV with comment text and 6 binary toxicity labels

### Download Dataset

```bash
# Using Kaggle API
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip
```

---

## 🎓 Model Training

### Training Pipeline Steps

1. **Data Preprocessing**
   - Load dataset from Cloud Storage
   - Balance toxic and non-toxic samples
   - Clean and normalize text

2. **Tokenization**
   - Tokenize comments using BERT tokenizer
   - Encode with `max_length=128`
   - Generate attention masks

3. **Model Training**
   - Fine-tune `bert-base-uncased` model
   - Multi-label classification with 6 classes
   - Training parameters:
     - Epochs: 5
     - Batch size: 32
     - Learning rate: 2e-5
     - Optimizer: AdamW

4. **Model Export**
   - Save model artifacts to Cloud Storage
   - Export `pytorch_model.bin` and `config.json`

### Running Training Pipeline

See `workflow.ipynb` for the complete Kubeflow Pipeline implementation.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex code
- Update documentation for new features
- Write tests for new functionality
- Ensure all tests pass before submitting

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: Authentication Error with GCP
**Solution**: Ensure service account JSON key is properly configured and has necessary permissions.

#### Issue: Docker Build Fails
**Solution**: Check Dockerfile syntax and ensure all dependencies are listed in `requirements.txt`.

#### Issue: Model Not Found Error
**Solution**: Verify model files (`pytorch_model.bin` and `config.json`) are in the `content/` directory.

#### Issue: Out of Memory Error
**Solution**: Increase Cloud Run memory allocation or reduce batch size during training.

#### Issue: Kaggle Dataset Download Fails
**Solution**: Verify `kaggle.json` credentials are correct and placed in `~/.kaggle/`.

### Getting Help

- Check the [Issues](https://github.com/parin117/Toxic-Comment-Classifier/issues) page
- Review `workflow.ipynb` for detailed setup instructions
- Consult [Google Cloud Documentation](https://cloud.google.com/docs)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and BERT models
- **Google Cloud Platform** for cloud infrastructure and services
- **Jigsaw/Conversation AI** for the toxic comment dataset
- **Kaggle** for hosting the competition and dataset
- **Gradio** for the intuitive UI framework
- **Google Gemini** for AI-powered comment stabilization

### References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 📧 Contact

**Parin Hirpara**

- **GitHub**: [@parin117](https://github.com/parin117)
- **Repository**: [Toxic-Comment-Classifier](https://github.com/parin117/Toxic-Comment-Classifier)

---

<div align="center">


[⬆ Back to Top](#-toxic-comment-classification--stabilization-system)

</div>

# 🛡️ Toxic Comment Classification & Stabilization 🚀

This project builds a **Toxic Comment Classifier** using a **BERT-based NLP model**, deployed on **Google Cloud Platform (GCP)** and integrated with a **Gradio UI** for real-time inference.

## 📌 Features
✅ Classifies user comments into **toxicity categories**.  
✅ Uses **BERT & Transformers** for NLP processing.  
✅ Deploys on **Google Cloud Run** via **Docker**.  
✅ **Stabilizes toxic comments** using **Google Gemini AI**.  
✅ Provides a **Gradio-based UI** for easy interaction.

## 🔧 Setup & Deployment
1. **Google Cloud Setup**
   - Create a **GCP account**, enable **Cloud Storage**, and set up a **new project**.
   - Store model artifacts in **Google Cloud Storage (GCS)**.
   
2. **Model Training**
   - **Fine-tune BERT** on a **toxic comment dataset**.
   - Train using **PyTorch & Hugging Face Transformers**.

3. **API Development**
   - Flask API for model inference.
   - Receives **POST requests**, processes text, and returns **toxicity classification**.

4. **Docker & Cloud Deployment**
   - Containerized using **Docker**.
   - Deployed on **Google Cloud Run**.
   - API available via **public endpoint**.

5. **Frontend (Gradio UI)**
   - Accepts user input and classifies toxicity.
   - If toxic, **stabilizes the comment** via **Google Gemini AI**.
   - Runs via **Gradio Web App**.

## 📖 Detailed Documentation
For a step-by-step guide, refer to the **detailed Jupyter Notebook**:  
[📄 workflow.ipynb](./workflow.ipynb)  
*(Ensure the notebook is in the same directory for the link to work.)*


🎯 **Built for AI-powered moderation & safer online interactions!** 🚀

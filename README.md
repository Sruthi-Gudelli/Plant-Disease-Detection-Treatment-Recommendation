### Plant Disease Detection & Treatment Recommendation System
A machine learning system that identifies plant leaf anomalies using Computer Vision and automatically generates targeted agricultural recovery protocols using a local Large Language Model (LLM).

---

### 📌 Project Overview
* **Stage 1 (Vision):** Comparative evaluation between a custom baseline Convolutional Neural Network (CNN) and a transfer-learning Residual Network (ResNet18) to classify plant diseases from leaf imagery.
* **Stage 2 (Language):** Integration of a locally hosted **qwen2.5:3b** model via Ollama to generate structured, actionable, and safe treatment recommendations based on the vision model's output.
* **Interface:** An interactive, localized Streamlit application built for real-time user image uploads, instant diagnosis, and localized cure protocols.

---

### 🛠️ Tech Stack & Key Libraries
* **Framework:** Streamlit (Frontend & Application State)
* **Deep Learning:** PyTorch & Torchvision (Model Training, Optimizations, & Visual Inference)
* **Local Inference Engine:** Ollama (Local Orchestration of LLM Context)
* **Language Model:** Qwen 2.5 (3B Parameters)

---

### 📂 Project Architecture & Workspace

├── Baseline_CNN_Model.ipynb                                        # Trained custom CNN
├── ResNet18_Model.ipynb                                            # Finetuned ResNet18 
├── Streamlit_Code.py                                               # Main Streamlit web application
├── requirements.txt                                                # Python dependencies
├── Plant Disease Detection and Treatment Recommendation.docx       # Comprehensive technical report
└── README.md                                                       # Project documentation


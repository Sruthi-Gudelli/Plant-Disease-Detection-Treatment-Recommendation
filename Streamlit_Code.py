import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import ollama


def Home_Page():
    st.title("🪴Plant Disease Detection and Treatment Recommendation")
    st.write("Plant diseases are a major threat to global agriculture, often causing severe losses in crop yield and quality. Many farmers struggle to identify diseases early, especially when symptoms are subtle or resemble nutrient deficiencies. Limited access to experts and timely diagnosis further complicates effective disease management.")
    st.write("This project aims to build an AI-driven system that can detect plant diseases directly from leaf images. It will classify the specific disease type using a trained deep-learning model. Then a Hugging Face Large Language Model will generate treatment recommendations. Therefore, this solution empowers farmers with quick, accurate, and actionable guidance for healthier crops.")
    st.subheader("⚙️Project Workflow:")
    st.markdown("""
    **Image Classification:** 
    * **Baseline CNN Model:** Firstly a baseline CNN model was built and trained on massive dataset. This is just to check whether the advanced model is doing better than this or not.
    * **ResNet18 Model:** A pre-trained ResNet18 model was taken and trained on massive dataset. This is used for real-time prediction.\n
    **Text Generation:**\n
    For treatment recommendation, **Qwen2.5-3B-Instruct** model is taken from Hugging Face.
    """)
    st.info("The detailed description regarding each model's performace is given in **Model Performance** page. Visit **Check Your Plant's Health** page to understand the health status of your plant.")


def Model_Performance():
    tab1, tab2, tab3 = st.tabs(['Baseline CNN Model', 'ResNet18 Model', 'Large Language Model'])
    with tab1:
        st.header("Baseline CNN Model", text_alignment='center')
        st.markdown("### 📊Performance at each epoch:")
        st.markdown("""
        | Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
        | :--- | :--- | :--- | :--- | :--- |
        | 1 | 1.1171 | 66.60% | 0.4010 | 87.13% |
        | 2 | 0.4491 | 85.60% | 0.2321 | 92.47% |
        | 3 | 0.2891 | 90.68% | 0.2121 | 92.88% |
        | 4 | 0.2152 | 92.92% | 0.1764 | 94.47% |
        | 5 | 0.1700 | 94.39% | 0.1626 | 94.84% |
        | 6 | 0.1402 | 95.35% | 0.1662 | 94.63% |
        | 7 | 0.1224 | 95.97% | 0.1821 | 94.80% |
        | 8 | 0.1006 | 96.69% | 0.1608 | 95.30% |
        | 9 | 0.0888 | 97.11% | 0.1737 | 95.37% |
        | 10 | 0.0855 | 97.26% | 0.1846 | 94.86% |
        """)

        st.write("\n")
        st.markdown("### 📋Additional Evaluation Metrics:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", value=0.9501)
        col2.metric("Recall", value=0.9486)
        col3.metric("F1 Score", value=0.9484)

        st.write("\n")
        st.markdown("### 🎯Predictions Made")
        st.write("To check how the model is performing, few images were given to it for prediction. The following are the predictions made by the model.")
        col4, col5 = st.columns(2, gap='small')
        col4.image('D:\VSCODE\FP_Plant_Disease_Detection\Baseline_image1.png')
        col5.image('D:\VSCODE\FP_Plant_Disease_Detection\Baseline_image2.png')
        col6, col7 = st.columns(2, gap='small')
        col6.image('D:\VSCODE\FP_Plant_Disease_Detection\Baseline_image3.png')
        col7.image('D:\VSCODE\FP_Plant_Disease_Detection\Baseline_image4.png')


    with tab2:
        st.header("ResNet18 Model", text_alignment='center')
        st.markdown("### 📊Performance at each epoch:")
        st.markdown("""
        | Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
        | :--- | :--- | :--- | :--- | :--- |
        | 1 | 0.1700 | 96.50% | 0.0448 | 98.74% |
        | 2 | 0.0172 | 99.59% | 0.0254 | 99.22% |
        | 3 | 0.0136 | 99.67% | 0.0175 | 99.53% |
        | 4 | 0.0110 | 99.72% | 0.0150 | 99.54% |
        | 5 | 0.0114 | 99.67% | 0.0168 | 99.45% |
        | 6 | 0.0082 | 99.78% | 0.0124 | 99.57% |
        | 7 | 0.0085 | 99.77% | 0.0183 | 99.45% |
        | 8 | 0.0054 | 99.86% | 0.0094 | 99.73% |
        | 9 | 0.0078 | 99.78% | 0.0147 | 99.55% |
        | 10 | 0.0060 | 99.82% | 0.0089 | 99.76% |
        """)

        st.write("\n")
        st.markdown("### 📋Additional Evaluation Metrics:")
        col1, col2, col3 = st.columns(3, gap='xxlarge')
        col1.metric("Precision", value=0.9976)
        col2.metric("Recall", value=0.9976)
        col3.metric("F1 Score", value=0.9976)

        st.write("\n")
        st.markdown("### 🎯Predictions Made")
        st.write("To check how the model is performing, few images were given to it for prediction. The following are the predictions made by the model.")
        col8, col9 = st.columns(2, gap='small')
        col8.image('D:\VSCODE\FP_Plant_Disease_Detection\ResNet_image1.png')
        col9.image('D:\VSCODE\FP_Plant_Disease_Detection\ResNet_image2.png')
        col10, col11 = st.columns(2, gap='small')
        col10.image('D:\VSCODE\FP_Plant_Disease_Detection\ResNet_image3.png')
        col11.image('D:\VSCODE\FP_Plant_Disease_Detection\ResNet_image4.png')

    with tab3:
        st.write("After CNN model predicted the disease, it is given to an LLM to generate:")
        st.markdown("""
        1. Disease description
        2. Causes
        3. Treatment recommendations
        4. Prevention tips
        """)
        st.write("There are many pre-trained LLM models available in Hugging Face.")
        st.info("**Qwen2.5-3B-Instruct** is taken for this project as it is completely open-access on Hugging Face and does not require any permission.")

def Prediction_Page():
    st.header("Plant Disease AI Doctor", text_alignment='center')
    st.write("To check whether your plant is healthy or having any disease, upload the image of your plant's leaf below. It will say whether the plant is healthy or not, if it is not healthy it will give you treatment recommendations and prevention tips.")
    
    @st.cache_resource
    def load_Resnet18_Model():
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 38) 
    
        weights = torch.load('D:\VSCODE\FP_Plant_Disease_Detection\ResNet18_Model_Weights.pth', map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        model.eval() 
        return model
    
    classifier = load_Resnet18_Model()
    # Exact alphabetical class list matching our training subfolder parameters
    CLASS_NAMES = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]

    resnet_preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    uploaded_image = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        raw_pil_image = Image.open(uploaded_image).convert('RGB')
        st.image(raw_pil_image, caption="Target Analysis Sample", use_container_width=True)
    
        input_tensor = resnet_preprocessing(raw_pil_image).unsqueeze(0)
        
        with torch.no_grad():
            logits = classifier(input_tensor)
            
            # Converts raw logits into probabilities (0.0 to 1.0) using Softmax
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Extracts the highest probability and its corresponding index position
            confidence_tensor, predicted_index = probabilities.max(1)
            
            # Converts the tensors to standard Python values
            confidence_score = confidence_tensor.item() * 100 
            raw_class_string = CLASS_NAMES[predicted_index.item()]
            
        # Cleans up class names (e.g., "Apple___Black_rot" -> "Apple", "Black rot")
        if "___" in raw_class_string:
            plant, disease = raw_class_string.split("___")
            disease = disease.replace("_", " ")
        else:
            plant, disease = "Unknown", raw_class_string
        
        # Displays the result along with the confidence score formatted to 2 decimal places
        st.success(f"🎯 **ResNet18 Identification:** {plant} — **{disease}** ({confidence_score:.2f}% Confidence)")
    
        if "healthy" in disease.lower():
            prompt = f"The {plant} crop sample has been diagnosed as healthy. Outline 1 direct management recommendations to maintain plant health and soil defenses."
        else:
            prompt = f"""Your task is to generate a highly actionable, structured treatment plan for a farmer whose crop has been diagnosed.
            Inputs:
            - Plant Type: {plant}
            - Diagnosis: {disease}
            Please structure your response with these exact sections:
            1. 📋 Summary of the Disease (What causes it?)
            2. 🛑 Immediate Action Steps (What should the farmer do today?)
            3. 🧪 Chemical/Organic Treatments (Fungicides, ratios, and safety protocols)
            4. 🌾 Future Prevention (How to keep the soil and next harvest clean)

            Keep the language universal, simple, and direct so it is easy for non-native speakers to understand. Avoid overly dense scientific jargon."""

        text_rendering_block = st.empty()
        accumulated_response_text = ""

        try:
            response = ollama.chat(
                model='qwen2.5:3b',
                messages=[
                    {'role': 'system', 'content': 'You are an expert AI Agronomist and Plant Pathologist.'},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True, 
                options={
                    "num_ctx": 1024 # Reduces the model context buffer from 4096+ down to 1024
                }
            )
            for chunk in response:
                accumulated_response_text += chunk['message']['content']
                # Re-render UI markdown on every newly calculated word
                text_rendering_block.markdown(accumulated_response_text)
            
                
        except Exception as network_error:
            st.error(f"Connection to Ollama Engine failed. Ensure 'ollama run qwen2.5:3b' remains active within your local command prompt window. Error logs: {network_error}")


    

home = st.Page(Home_Page, title='Home')
performance = st.Page(Model_Performance, title='Model Performance')
prediction = st.Page(Prediction_Page, title="🩺Check Your Plant's Health")

pg = st.navigation([home, performance, prediction])
pg.run()
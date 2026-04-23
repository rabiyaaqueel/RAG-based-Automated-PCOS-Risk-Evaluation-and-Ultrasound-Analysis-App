import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
import os
from PIL import Image
import io
import faiss
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime
from tensorflow.keras.models import load_model

#Configuration & Models
NVIDIA_API_KEY = "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" 
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

#GLOBAL CONFIGURATION 
st.set_page_config(page_title="RAG PCOS", page_icon="🩺", layout="centered")

#SHARED ASSET LOADING
@st.cache_resource
def load_symptom_models():
    # Meta-Learner and Base Models
    meta_model = joblib.load("final_stacked_model_GB.pkl")
    nn_model = tf.keras.models.load_model("base_nn.keras")
    rf_model = joblib.load("base_rf.pkl")
    lr_model = joblib.load("base_lr.pkl")
    xgb_model = joblib.load("base_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    return scaler, nn_model, rf_model, lr_model, xgb_model, meta_model

@st.cache_resource
def load_ultrasound_model():
    # Replace with your actual ultrasound model filename
    return tf.keras.models.load_model("pcos_ultrasound_model.keras")

@st.cache_resource
def load_rag_models():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_rag_models()


#RAG Helper Functions

def load_csv_rag(uploaded_file):
    df = pd.read_csv(uploaded_file)
    text_data = []
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        text_data.append(row_text)
    return df, text_data

def build_index_rag(texts):
    embeddings = embed_model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def ask_nvidia_rag(context, question):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.1-8b-instruct", 
        "messages": [
            {
                "role": "system",
                "content": "You are a professional medical assistant specialized in PCOS data analysis. Use the provided context to answer questions accurately."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}"
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    try:
        res = requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload, timeout=20)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API Error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"


# Initialize session_state keys if they don't exist
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "rag_doc_chunks" not in st.session_state:
    st.session_state.rag_doc_chunks = []
if "rag_df_preview" not in st.session_state:
    st.session_state.rag_df_preview = None
if "rag_initial_summary" not in st.session_state:
    st.session_state.rag_initial_summary = None

#pages
page = st.sidebar.radio("Pages", ["RAB PCOS", "PCOS Prediction (Symptoms)", "Ultrasound Prediction","RAG PCOS Symptoms Analysis","RAG Chatbot- Your personal medical assistant"])

# home page
if page == "RAB PCOS":
    st.title("RAB PCOS")
    st.subheader("Automated PCOS Risk Evaluation and Ultrasound Analysis App")
    st.markdown("""
    ### Welcome 👋
    
    ### Overview
    The RAB PCOS platform is an integrated medical intelligence solution designed to assist healthcare providers 
    and patients in the early detection and management of Polycystic Ovary Syndrome. By combining traditional 
    clinical data with state-of-the-art Computer Vision and Retrieval-Augmented Generation (RAG), 
    the app provides a 360-degree diagnostic view.
    
    ### How to Use the App
    Step 1: Select PCOS Prediction from the sidebar to enter clinical measurements (Weight, Cycle Length, etc.).

    Step 2: Navigate to Ultrasound Prediction to upload and analyze medical imaging.

    Step 3: Use RAG Analysis to upload a patient history CSV and get an AI-generated health summary.

    Step 4: Consult the RAG Chatbot for specific questions regarding the uploaded data.
    """)

# pcos pred page
elif page == "PCOS Prediction (Symptoms)":
    st.title("PCOS Prediction Based on Symptoms")
    
    try:
        scaler, nn_model, rf_model, lr_model, xgb_model, meta_model = load_symptom_models()
        
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Numeric Measurements")
                weight = st.number_input("Weight (Kg)", value=60.0)
                height = st.number_input("Height (Cm)", value=160.0)
                age = st.number_input("Age (yrs)", value=25)
                cycle_len = st.number_input("Cycle Length (days)", value=28)
            
            with col2:
                st.subheader("Symptoms & Habits")
                s_dark = st.selectbox("Skin Darkening", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                h_grow = st.selectbox("Hair Growth", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                w_gain = st.selectbox("Weight Gain", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                cycle_choice = st.selectbox("Cycle Type", ["Regular", "Irregular"])
                c_irr = 2 if cycle_choice == "Regular" else 4
                fast_food = st.selectbox("Fast Food", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                pimp = st.selectbox("Pimples", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                h_loss = st.selectbox("Hair Loss", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                exer = st.selectbox("Regular Exercise", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

            submit = st.form_submit_button("Run Analysis")

        if submit:
            # Predict
            X_cont_scaled = scaler.transform(np.array([[weight, height, age, cycle_len]]))
            X_bin = np.array([[s_dark, h_grow, w_gain, c_irr, fast_food, pimp, h_loss, exer]])
            X_combined = np.hstack([X_cont_scaled, X_bin])
            
            prob_nn = nn_model.predict([X_cont_scaled, X_bin], verbose=0).flatten()
            prob_rf = rf_model.predict_proba(X_combined)[:, 1]
            prob_lr = lr_model.predict_proba(X_combined)[:, 1]
            prob_xgb = xgb_model.predict_proba(X_combined)[:, 1]
            
            X_meta = np.column_stack([prob_nn, prob_rf, prob_lr, prob_xgb])
            final_prob = meta_model.predict_proba(X_meta)[0][1]

            # Logic for Risk Levels
            st.divider()
            if final_prob < 0.35:
                risk_lvl = "Low Risk"
                st.success(f"### Assessment: {risk_lvl}")
            elif final_prob < 0.70:
                risk_lvl = "Medium Risk"
                st.warning(f"### Assessment: {risk_lvl}")
                st.info("⚠ Please consult a doctor and proceed to the Ultrasound Scan module for further verification.")
            else:
                risk_lvl = "High Risk"
                st.error(f"### Assessment: {risk_lvl}")
                st.info("🚨 Strong symptoms detected. Immediate consultation required. Please perform an Ultrasound Scan.")

            st.write(f"**Confidence Score:** {final_prob:.2%}")


            # Data Preparation for Download
            current_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Age": age, "Weight": weight, "Height": height, "Cycle_Len": cycle_len,
                "Cycle_Type": cycle_choice, "Skin_Darkening": s_dark, "Hair_Growth": h_grow,
                "Weight_Gain": w_gain, "Fast_Food": fast_food, "Pimples": pimp,
                "Hair_Loss": h_loss, "Exercise": exer, "Probability": round(final_prob, 4),
                "Risk_Level": risk_lvl
            }
            df_download = pd.DataFrame([current_data])
            
            # Download Button
            csv_buffer = io.StringIO()
            df_download.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download My Prediction Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"PCOS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")

# ULTRASOUND PREDICTION Page
elif page == "Ultrasound Prediction":
    st.title("PCOS Prediction Ultrasound Scan(CNN)")
    st.markdown("Prediction of PCOS based on ovary ultrasoundscan")
    st.write("Upload a scan to check for PCOS infection status.")

    # 1. Class Mapping Logic
    # Update these based on your test_gen.class_indices from the training log!
    CLASS_MAP = {0: "Infected", 1: "Not Infected"}

    # 2. Load Model from Local Path
    # To this:
    model = load_model('best_model.keras')

    @st.cache_resource
    def load_cnn_model(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    u_model = load_cnn_model(model)

    # 3. File Uploader
    uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_container_width=True)
        
        if st.button("Analyze Scan"):
            if u_model is not None:
                with st.spinner("Neural Network analyzing image..."):
                    try:
                        # --- 4. PREPROCESSING ---
                        img = image.convert("RGB")
                        img = img.resize((224, 224))
                        img_array = np.array(img).astype('float32') / 255.0
                        img_tensor = np.expand_dims(img_array, axis=0)
                        
                        # --- 5. PREDICTION LOGIC ---
                        prediction_prob = u_model.predict(img_tensor)[0][0]
                        
                        # Apply your training threshold (0.70)
                        # If probability > 0.7, it is Class 1. Otherwise Class 0.
                        predicted_class_idx = 1 if prediction_prob > 0.70 else 0
                        result_label = CLASS_MAP[predicted_class_idx]
                        
                        st.divider()
                        
                        # --- 6. DISPLAY RESULTS ---
                        if result_label == "Infected":
                            st.error(f"### Assessment: {result_label}")
                            # If Infected is Class 0, confidence is (1 - prob)
                            # If Infected is Class 1, confidence is prob
                            conf = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)
                            st.write(f"**PCOS Probability:** {conf:.2%}")
                            st.warning("🚨 Indicators of PCOS detected. Consult a physician.")
                        else:
                            st.success(f"### Assessment: {result_label}")
                            conf = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)
                            st.write(f"**Confidence Score:** {conf:.2%}")
                            st.info("✅ No significant indicators of PCOS detected.")

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            else:
                st.error("Model file not found. Check the MODEL_PATH.")
elif page == "RAG PCOS Symptoms Analysis":
    # Tab 3: RAG PCOS Symptoms Analysis
    st.title("RAG PCOS Symptoms Analysis")
    st.markdown("### Upload patient records for AI-powered insights.")

    rag_file = st.file_uploader("Upload PCOS CSV File", type=["csv"])

    if rag_file:
        if st.button("Analyze & Index Data"):
            with st.spinner("Processing medical records..."):
                df, texts = load_csv_rag(rag_file)
                st.session_state.rag_doc_chunks = texts
                st.session_state.rag_index = build_index_rag(texts)
                st.session_state.rag_df_preview = df.head()
                
                # Generate initial summary
                sample_context = " ".join(texts[:5]) 
                st.session_state.rag_initial_summary = ask_nvidia_rag(sample_context, "Provide a summary of the patient health trends in this data.")
                
                st.success("Data indexed successfully! You can now use the Chatbot tab for specific queries.")

        # Display results if data is processed
        if st.session_state.rag_df_preview is not None:
            st.write("#### CSV Preview (Top 5 Rows)")
            st.dataframe(st.session_state.rag_df_preview)

        if st.session_state.rag_initial_summary:
            st.markdown("### 📋 AI Health Summary")
            st.info(st.session_state.rag_initial_summary)
    else:
        st.info("Please upload a CSV file to begin analysis.")
elif page == "RAG Chatbot- Your personal medical assistant":
    st.title("🤖 RAG Chatbot")
    st.subheader("Your Personal Medical Assistant")

    # Change 1: Show the input field regardless of CSV status
    st.markdown("### 💬 Ask any questions about PCOS")
    user_query = st.text_input("Ask about PCOS general info or your uploaded records:", key="chat_input")

    if user_query:
        with st.spinner("Generating response..."):
            # Change 2: Check if we have CSV context to add
            if st.session_state.rag_index is not None:
                # Retrieve context from CSV
                q_emb = embed_model.encode([user_query]).astype("float32")
                D, I = st.session_state.rag_index.search(q_emb, k=3)
                context = " ".join([st.session_state.rag_doc_chunks[i] for i in I[0]])
                
                # Ask with context
                answer = ask_nvidia_rag(context, user_query)
            else:
                # Change 3: Ask as a general medical assistant (No context)
                # We pass an empty string or a general prompt as context
                general_context = "No specific patient data uploaded. Answer based on general medical knowledge of PCOS."
                answer = ask_nvidia_rag(general_context, user_query)

            st.markdown("#### 🤖 AI Assistant:")
            st.write(answer)
            
    # Optional: Small note to let user know data isn't loaded
    if st.session_state.rag_index is None:
        st.caption("💡 Note: You haven't uploaded a CSV yet. I am answering based on general medical knowledge.")
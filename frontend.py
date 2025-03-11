import streamlit as st
import os
from backend import process_files, cleanup_temp_files
from PIL import Image

# Streamlit frontend for uploading PDFs and processing them
st.title("PDF to Image Conversion, OCR, and Model Processing")
st.markdown("""
    Upload PDFs to convert them to images, run OCR on the images to extract text, and process the text with a model.
""")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# If files are uploaded
if uploaded_files:
    st.write(f"Number of PDFs uploaded: {len(uploaded_files)}")

    # Process the uploaded files (convert PDF to images, run OCR, call Groq API)
    with st.spinner("Converting PDFs to images and running OCR..."):
        processed_images, extracted_texts, model_outputs = process_files(uploaded_files)

    # Display processed images
    st.subheader("Processed Images:")
    for img_path in processed_images:
        img = Image.open(img_path)
        st.image(img, caption=os.path.basename(img_path), use_column_width=True)

    # Display extracted text from OCR
    st.subheader("Extracted Text from OCR:")
    for text in extracted_texts:
        st.text_area("OCR Output", value=text, height=150)

    # Display the extracted model output
    st.subheader("Model Output:")
    st.json(model_outputs)

    # Clean up temporary files after processing
    cleanup_temp_files()

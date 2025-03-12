import os
import uuid
import shutil
import cv2
import fitz  # PyMuPDF
import requests
import time
import json
from paddleocr import PaddleOCR
from PIL import Image
import re
import requests
from dotenv import load_dotenv

# Create directories for temporary files
temp_dir = './temp_files'
os.makedirs(temp_dir, exist_ok=True)

# Directories for processed images and text output
output_folder_pdftoimage = './temp_files/images'
os.makedirs(output_folder_pdftoimage, exist_ok=True)
output_folder_imagepreprocessed = './temp_files/processed_images'
os.makedirs(output_folder_imagepreprocessed, exist_ok=True)
text_output_folder = './temp_files/extracted_text'
os.makedirs(text_output_folder, exist_ok=True)
json_output_folder = './temp_files/extracted_json'
os.makedirs(json_output_folder, exist_ok=True)

# Load the .env file
load_dotenv()

# Access environment variables
groq_api_url = os.getenv("GROQ_API_URL")
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")

# Initialize PaddleOCR with structure mode enabled
ocr = PaddleOCR(
    lang="en",
    det=True,
    rec=True,
    cls=True,
    use_angle_cls=True,
    structure=True,  #  Enables structure recognition for better text extraction
    det_db_box_thresh=0.5,
    rec_algorithm='SVTR_LCNet',
    drop_score=0.75  # Filters low-confidence text
)

# Backend functions

def pdfs_to_first_images(pdf_paths, output_folder=output_folder_pdftoimage):
    image_paths = []
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            first_page = doc[0]  # Extract first page
            pix = first_page.get_pixmap(dpi=300)  # High-quality image

            img_filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_1.png"
            img_path = os.path.join(output_folder, img_filename)

            pix.save(img_path)
            image_paths.append(img_path)

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return image_paths

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #contrast_enhanced = clahe.apply(gray)
    #normalized = cv2.normalize(contrast_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    #denoised = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)
    #binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    corrected = correct_rotation(gray)

    cv2.imwrite(output_path, corrected)
    return output_path

def extract_text_from_images(input_folder, text_output_folder):
    for image_name in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_name)
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
            continue

        try:
            result = ocr.ocr(image_path, cls=True)
            extracted_data = []
            text_lines = []
            for line in result[0]:
                if line[1] and line[1][1] >= 0.75:
                    cleaned_text = clean_ocr_text(line[1][0])
                    extracted_data.append({"text": cleaned_text, "confidence": round(line[1][1], 4)})
                    text_lines.append(cleaned_text)

            if not extracted_data:
                continue

            text_filename = os.path.splitext(image_name)[0] + "_extracted.txt"
            text_file_path = os.path.join(text_output_folder, text_filename)

            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

def classify_document(ocr_text):
    """Classifies document into Utilities, Telcom, or Credit Cards based on OCR content."""

    # Convert text to lowercase for case-insensitive matching
    ocr_text_lower = ocr_text.lower()

    # ✅ Fix: Replace multiple spaces/newlines to join split words correctly
    cleaned_text = re.sub(r'\s+', ' ', ocr_text_lower)  # Normalize text spacing

    # ✅ Category: Credit Card Statements (Check for Bank Names & Key Financial Terms)
    credit_card_keywords = [
        "credit card", "payment due date", "new balance", "minimum payment due",
        "total amount due", "credit limit", "available credit", "statement date"
    ]

    # ✅ Fix: Use regex to catch split words for "Standard Chartered"
    bank_patterns = [
        r"standard\s+chartered", r"scb", r"ocbc", r"dbs", r"citibank", r"uob",
        r"hsbc", r"maybank", r"american express", r"personal loan statement"
    ]

    # If a bank name is found & financial terms are present → It's a Credit Card Statement
    if any(re.search(bank, cleaned_text) for bank in bank_patterns) and any(term in cleaned_text for term in credit_card_keywords):
        return "Credit_Card"

    # ✅ Category: Telcom Bills (Match Telecom Provider Names & Keywords)
    telco_providers = ["singtel", "starhub", "m1", "tpg"]
    telco_keywords = ["billing period", "mobile plan", "data usage", "broadband", "internet bill", "bill date", "invoice date", "total due"]

    if any(telco in cleaned_text for telco in telco_providers) and any(term in cleaned_text for term in telco_keywords):
        return "Telcom"

    # ✅ Category: Utilities Bills (Electricity, Water, Gas)
    utilities_providers = ["sp services", "pub", "senoko", "tengah", "keppel electric"]
    utilities_keywords = ["electricity charges", "water charges", "gas charges", "utility bill", "meter reading"]

    if any(util in cleaned_text for util in utilities_providers) or any(term in cleaned_text for term in utilities_keywords):
        return "Utilities"

    # ❌ If No Match, Return "Unknown"
    return "Unknown_Document"

def get_extraction_prompt(document_type, ocr_text):
    """
    Returns the appropriate extraction prompt based on the document classification.
    If the document is a Credit Card statement and the bank is OCBC, it applies the `extract_ocbc_fields` function first.
    """

    # ✅ If the document is a credit card statement
    if document_type == "Credit_Card":
        return f"""
        You are a financial document extraction assistant. Extract key financial fields and return ONLY valid JSON.

        **STRICT OUTPUT FORMAT RULES:**
        - **Return only raw JSON. DO NOT include any explanations, formatting, or markdown code blocks.**
        - **DO NOT include text like "Here is the extracted JSON" or "Output:".**
        - **Ensure numeric values are extracted correctly** (e.g., 2730.15).
        - **If any value is missing, return `"Not Available"`**.

        ### **Bank-Specific Adjustments**
        - **Standard Chartered / Citibank / DBS / UOB / Maybank:**
          - **"Total Amount Due" may be labeled as:**
            - **"Statement Balance"**
            - **"New Balance"**
          - **Prioritize this field** and ignore "Subtotal".
          - **DO NOT use "Transaction Summary" or "Previous Balance".**
          - **If the account number format is like "5498-34XX-XXXX-7348", detect it as the account number.**

        **Final JSON Output Format:**
        ```json
        {{
            "Bank Name": "Extracted Bank Name",
            "Account Holder Name": "Extracted Full Name",
            "Account Number": "Extracted Account Number",
            "Statement Date": "YYYY-MM-DD",
            "Payment Due Date": "YYYY-MM-DD or Not Available",
            "Total Amount Due": 0.00,
            "Minimum Payment Due": 0.00,
            "Credit Limit": 0.00
        }}
        ```

        **OCR Text for Extraction:**
        ```
        {ocr_text}
        ```
        """

    # ✅ If it's a Telecom Bill
    elif document_type == "Telcom":
        return f"""
        You are a highly accurate document extraction assistant. Extract structured key fields from the given OCR text and return ONLY valid JSON.

        **Rules:**
        - DO NOT include extra text or explanations.
        - Ensure currency values are formatted correctly (e.g., $203.69).
        - If a value is missing, set it to `"Not Available"`.
        - Extract only relevant fields for a **telecom bill**.
        - If the provider name is **"Singapore Telecommunications Limited"**, return **"Singtel"** instead.
        - If the provider name is **"StarHub Ltd"**, return **"StarHub"** instead.
        - If the provider name is **"M1 Limited"**, return **"M1"** instead.
        - **DO NOT return text explanations**—only the JSON output.

        **Final JSON Output Format:**
        ```json
        {{
            "Provider Name": "Extracted Telecom Company",
            "Account Holder Name": "Extracted Full Name",
            "Account Number": "Extracted Account Number",
            "Billing Period": "Start Date - End Date",
            "Bill Date": "YYYY-MM-DD",
            "Total Amount Due": 0.00,
            "Due Date": "YYYY-MM-DD"
        }}
        ```

        **OCR Text for Extraction:**
        ```
        {ocr_text}
        ```
        """

    # ✅ If it's a Utilities Bill
    elif document_type == "Utilities":
        return f"""
        You are a structured document extraction assistant. Extract key fields from the given OCR text and return ONLY valid JSON.

        **Rules:**
        - DO NOT include extra text or explanations.
        - Ensure currency values are formatted correctly (e.g., $51.57).
        - If a value is missing, set it to `"Not Available"`.
        - Extract only relevant fields for a **utilities bill**.
        - **DO NOT return text explanations**—only the JSON output.

        **Final JSON Output Format:**
        ```json
        {{
            "Provider Name": "Extracted Utility Provider",
            "Account Holder Name": "Extracted Full Name",
            "Account Number": "Extracted Account Number",
            "Billing Period": "Start Date - End Date",
            "Bill Date": "YYYY-MM-DD",
            "Total Amount Due": 0.00,
            "Due Date": "YYYY-MM-DD",
            "Electricity Charges ($)": 0.00,
            "Water Charges ($)": 0.00,
            "Gas Charges ($)": 0.00
        }}
        ```

        **OCR Text for Extraction:**
        ```
        {ocr_text}
        ```
        """

    # ❌ If Document Type is Not Recognized
    else:
        return "Document type not recognized. Please check OCR text."

def call_groq_api(prompt):
    """
    Sends a request to the Groq API and handles rate limits.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a document extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1000  # Adjusted to prevent exceeding token limit
    }

    # Retry logic in case of rate limits
    while True:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            # Parse the response JSON and return the content
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")

        elif response.status_code == 429:
            # Handle rate limit by extracting the wait time from the error message
            error_data = response.json()
            wait_time = float(error_data.get("error", {}).get("message").split("Please try again in ")[1].split("ms")[0]) / 1000
            print(f"⚠️ Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait and retry
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None

# Function to process PDFs to images and extract text
def convert_pdf_to_text(pdf_files):
    processed_image_paths = []  # Store processed image paths
    extracted_texts = []  # Store extracted texts from OCR

    # Iterate through the uploaded PDF files
    for pdf_file in pdf_files:
        unique_pdf_filename = f"/content/temp_uploaded_pdf_{str(uuid.uuid4())}.pdf"
        with open(unique_pdf_filename, "wb") as f:
            f.write(pdf_file)  # Save the uploaded PDF to a local file

        # Convert the first page to image and get the path
        image_paths = pdfs_to_first_images([unique_pdf_filename], output_folder=output_folder_pdftoimage)

        # Preprocess the images and extract text
        if image_paths:
            processed_images = []
            for image_path in image_paths:
                processed_image = preprocess_image(image_path, output_path=os.path.join(output_folder_imagepreprocessed, os.path.basename(image_path)))
                processed_images.append(processed_image)

            processed_image_paths.extend(processed_images)

            # Extract text from the images
            extract_text_from_images(output_folder_imagepreprocessed, text_output_folder, json_output_folder)

            # Collect extracted texts, ensuring the text files exist
            for image_path in processed_images:
                extracted_text_file = os.path.join(text_output_folder, os.path.basename(image_path).replace(".png", "_extracted.txt"))
                if os.path.exists(extracted_text_file):
                    with open(extracted_text_file, "r") as f:
                        extracted_text = f.read()
                        extracted_texts.append(extracted_text)
                else:
                    print(f"⚠️ Text extraction file for {image_path} not found.")
                    extracted_texts.append("Text extraction failed.")

    return processed_image_paths, extracted_texts

def run_model_on_extracted_texts(extracted_texts):
    """
    Run the model on the extracted text from OCR.
    """
    extracted_jsons = []  # Store the final extracted JSON results

    # Process the extracted texts and call the model API
    for ocr_text in extracted_texts:
        document_type = classify_document(ocr_text)
        prompt = get_extraction_prompt(document_type, ocr_text)

        # Call the model API with retry logic
        extracted_json = call_groq_api(prompt)

        # Save output if valid
        if extracted_json:
            extracted_jsons.append(extracted_json)

    return extracted_jsons

def process_files(pdf_files):
    processed_images, extracted_texts = convert_pdf_to_text(pdf_files)
    model_outputs = run_model_on_extracted_texts(extracted_texts)
    return processed_images, extracted_texts, model_outputs

# Clean up temporary files after processing
def cleanup_temp_files():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

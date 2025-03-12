# PDF Document Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit application that extracts structured information from PDF documents (bills, statements, etc.) using OCR and large language models.

## 🚀 Features

- **PDF Processing**: Converts PDF first pages to high-resolution images
- **OCR Text Extraction**: Uses PaddleOCR for accurate text recognition
- **Smart Document Classification**: Automatically categorizes documents as:
  - Credit Card Statements
  - Telecom Bills
  - Utility Bills
- **Structured Data Extraction**: Uses Groq LLM API to eact key fields in JSON format
- **Clean UI**: Simple Streamlit interface for uploading and viewing results

## 📋 Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyMuPDF
- PaddleOCR
- Pillow
- Requests

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/AbdulRauf2/fin_doc_extract
cd fin_doc_extract
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your Groq API key:
   - Create a Groq account and generate an API key
   - Set the key in `backend.py` or use environment variables

## 💻 Usage

1. Start the application:
```bash
streamlit run frontend.py
```

2. Upload PDF documents through the file uploader

3. View the processing results:
   - Converted document images
   - Extracted OCR text
   - Structured JSON data with document fields

## 🔄 How It Works

```
PDF Upload → Image Conversion → Preprocessing → OCR → Classification → Data Extraction → JSON Output
```

The system processes documents through this pipeline:
1. Extracts the first page from each PDF at 300 DPI
2. Preprocesses images for optimal OCR accuracy
3. Runs PaddleOCR to extract text content
4. Classifies document type based on text patterns
5. Generates tailored prompts for the Groq LLM API
6. Returns structured JSON with key document fields

## 📄 Supported Document Types

### Credit Card Statements
```json
{
    "Bank Name": "Bank Name",
    "Account Holder Name": "Full Name",
    "Account Number": "Account Number",
    "Statement Date": "YYYY-MM-DD",
    "Payment Due Date": "YYYY-MM-DD",
    "Total Amount Due": 0.00,
    "Minimum Payment Due": 0.00,
    "Credit Limit": 0.00
}
```

### Telecom Bills
```json
{
    "Provider Name": "Telecom Company",
    "Account Holder Name": "Full Name",
    "Account Number": "Account Number",
    "Billing Period": "Start Date - End Date",
    "Bill Date": "YYYY-MM-DD",
    "Total Amount Due": 0.00,
    "Due Date": "YYYY-MM-DD"
}
```

### Utility Bills
```json
{
    "Provider Name": "Utility Provider",
    "Account Holder Name": "Full Name",
    "Account Number": "Account Number",
    "Billing Period": "Start Date - End Date",
    "Bill Date": "YYYY-MM-DD",
    "Total Amount Due": 0.00,
    "Due Date": "YYYY-MM-DD",
    "Electricity Charges ($)": 0.00,
    "Water Charges ($)": 0.00,
    "Gas Charges ($)": 0.00
}
```

## ⚠️ Limitations

- Currently processes only the first page of each PDF
- OCR accuracy depends on document quality and formatting
- API rate limits may affect processing speed for large batches

## 🔮 Future Improvements

- [ ] Multi-page PDF processing
- [ ] Support for additional document types (invoices, receipts, etc.)
- [ ] Local model option to reduce API dependency
- [ ] Document history and comparison features
- [ ] Enhanced validation and error handling

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

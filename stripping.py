import pdfplumber

def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # skip blank pages
                text += page_text + "\n"
    return text

# Usage:
text = extract_text_pdfplumber("coproprio_fr.pdf")

# Fix encoding artifacts
import unicodedata

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")

cleaned_text = clean_text(text)

# Save the extracted text to a txt file
output_file = "extracted_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

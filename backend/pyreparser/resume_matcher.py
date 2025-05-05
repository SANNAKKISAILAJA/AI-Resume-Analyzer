import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_name_and_email(text):
    # Extract email
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    
    # Try labeled name extraction first
    name_match = re.search(r"(?:Name|Full Name)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    
    if name_match:
        name = name_match.group(1).strip()
    else:
        # Fallback to top of resume: first meaningful line
        lines = text.splitlines()
        for line in lines:
            if line.strip() and any(c.isalpha() for c in line) and len(line.strip().split()) <= 5:
                name = line.strip()
                break
        else:
            name = "Unknown"
    
    return name, email.group(0) if email else "Not found"

def match_resumes(job_description, resumes, candidate_count):
    texts = [job_description]
    resume_data = []

    for resume in resumes:
        content = resume["content"]
        text = extract_text_from_pdf(content)
        pdf_name = resume["filename"]
        name, email = extract_name_and_email(text)
        resume_data.append({ "name": name, "email": email,  "pdf_name": pdf_name, "text": text})
        texts.append(text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(scores):
        resume_data[i]["score"] = round(float(score) * 100, 2)

    resume_data.sort(key=lambda x: x["score"], reverse=True)
    return resume_data[:candidate_count]

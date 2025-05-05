import csv
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from resume_matcher import extract_text_from_pdf, extract_name_and_email
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5178"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_resumes(
    job_description: str = Form(...),
    candidate_count: int = Form(...),
    resumes: List[UploadFile] = File(...),
):
    texts = [job_description]
    resume_data = []

    for resume in resumes:
        content = await resume.read()
        text = extract_text_from_pdf(content)
        name, email = extract_name_and_email(text)
        
        # Capture the filename of the PDF
        pdf_name = resume.filename  # This will give the name of the uploaded PDF file
        
        # Store resume data, including the filename and status (default as Pending)
        resume_data.append({
            "name": name,
            "email": email,
            "pdf_name": pdf_name,  # Include pdf_name in the data
            "score": 0,  # Default score
            "status": "Pending"  # Default status is "Pending"
        })

        texts.append(text)

    # TF-IDF Matching
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(scores):
        resume_data[i]["score"] = round(float(score) * 100, 2)  # Convert to percentage

    # Sort and select top N candidates
    resume_data.sort(key=lambda x: x["score"], reverse=True)
    shortlisted = resume_data[:candidate_count]

    # Save the shortlisted candidates to CSV
    save_to_csv(shortlisted)

    # Return the shortlisted candidates as JSON
    return shortlisted


# Function to save shortlisted candidates to CSV
def save_to_csv(shortlisted):
    with open('shortlisted_candidates.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "email", "pdf_name", "score", "status"])
        if file.tell() == 0:  # If the file is empty, write the header
            writer.writeheader()
        for candidate in shortlisted:
            writer.writerow(candidate)


@app.post("/accept_candidate")
async def accept_candidate(candidate: dict):
    # Open the CSV file and update the status
    candidates = []

    with open('shortlisted_candidates.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["email"] == candidate["email"]:
                row["status"] = "Accepted"  # Update the status of the candidate
            candidates.append(row)

    # Write the updated data back to the CSV file
    with open('shortlisted_candidates.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "email", "pdf_name", "score", "status"])
        writer.writeheader()
        writer.writerows(candidates)

    return {"message": f"Candidate {candidate['name']} has been accepted."}

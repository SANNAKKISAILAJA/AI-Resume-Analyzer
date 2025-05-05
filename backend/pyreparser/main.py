import os
import csv
from tempfile import NamedTemporaryFile
from typing import List

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from resume_parser import ResumeParser  # Local parser
from resume_matcher import extract_text_from_pdf, extract_name_and_email
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Allow frontend requests (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5177", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Video Recommendations and Role Prediction ---

video_recommendations = {
    "Data Scientist": "https://www.youtube.com/watch?v=YJZCUhxNCv8",
    "Web Developer": "https://www.youtube.com/watch?v=Zr6r3D8QTPY",
    "Android Developer": "https://www.youtube.com/watch?v=_XJ6QTlrE94",
    "UI/UX Designer": "https://www.youtube.com/watch?v=Q7AOvWpIVHU",
    "Software Engineer": "https://www.youtube.com/watch?v=pzK0sRJH08E",
    "Machine Learning Engineer": "https://www.youtube.com/watch?v=GxZrEKZfWmY"
}

def predict_role(skills):
    skill_map = {
        "data": "Data Scientist",
        "python": "Software Engineer",
        "html": "Web Developer",
        "android": "Android Developer",
        "ux": "UI/UX Designer",
        "machine": "Machine Learning Engineer",
    }
    for skill in skills:
        for keyword, role in skill_map.items():
            if keyword in skill.lower():
                return role
    return "Software Engineer"

# --- Resume Parsing Route ---

@app.post("/parse-resume")
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        with NamedTemporaryFile(delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        parser = ResumeParser(temp_path)
        data = parser.get_extracted_data()

        os.remove(temp_path)

        skills = data.get("skills", [])
        predicted_role = predict_role(skills)
        video = video_recommendations.get(predicted_role, "No video available")

        return {
            "skills": skills,
            "score": data.get("score", 0),
            "improvements": data.get("improvements", []),
            "predicted_role": predicted_role,
            "video_recommendation": video,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Resume Upload and Matching Route ---

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

        resume_data.append({
            "name": name,
            "email": email,
            "pdf_name": resume.filename,
            "score": 0,
            "status": "Pending"
        })

        texts.append(text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(scores):
        resume_data[i]["score"] = round(float(score) * 100, 2)

    resume_data.sort(key=lambda x: x["score"], reverse=True)
    shortlisted = resume_data[:candidate_count]

    save_to_csv(shortlisted)

    return shortlisted

def save_to_csv(shortlisted):
    with open('shortlisted_candidates.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "email", "pdf_name", "score", "status"])
        if file.tell() == 0:
            writer.writeheader()
        for candidate in shortlisted:
            writer.writerow(candidate)

# --- Accept Candidate Route ---

@app.post("/accept_candidate")
async def accept_candidate(candidate: dict):
    candidates = []

    with open('shortlisted_candidates.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["email"] == candidate["email"]:
                row["status"] = "Accepted"
            candidates.append(row)

    with open('shortlisted_candidates.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "email", "pdf_name", "score", "status"])
        writer.writeheader()
        writer.writerows(candidates)

    return {"message": f"Candidate {candidate['name']} has been accepted."}

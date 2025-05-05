import React, { useState } from "react";
import "./Recruiterdashboard.css";
import axios from "axios";

const Recruiterdashboard = () => {
  const [jobDesc, setJobDesc] = useState("");
  const [resumes, setResumes] = useState([]);
  const [count, setCount] = useState(1);
  const [shortlisted, setShortlisted] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("job_description", jobDesc);
    formData.append("candidate_count", count);

    for (let i = 0; i < resumes.length; i++) {
      formData.append("resumes", resumes[i]);
    }

    try {
      const res = await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setShortlisted(res.data);
    } catch (error) {
      alert("Error: " + error.message);
    }
  };

  const handleAccept = async (candidate) => {
    try {
      // Send the accepted candidate to the backend to store in CSV
      await axios.post("http://127.0.0.1:8000/accept_candidate", candidate);
      alert(`Candidate ${candidate.name} added successfully to CSV`);
    } catch (error) {
      alert("Error: " + error.message);
    }
  };

  const handleReject = (candidate) => {
    alert(`Application from ${candidate.name} has been rejected.`);
  };

  return (
    <div className="form-container">
      <h2>Smart Resume Shortlisting</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter Job Description"
          value={jobDesc}
          onChange={(e) => setJobDesc(e.target.value)}
          rows="6"
        />
        <input
          type="file"
          multiple
          onChange={(e) => setResumes(Array.from(e.target.files))
          }

        />
        <input
          type="number"
          value={count}
          onChange={(e) => setCount(e.target.value)}
          min="1"
        />
        <button type="submit">Shortlist Candidates</button>
      </form>

      {shortlisted.length > 0 && (
        <div className="results">
          <h3>Shortlisted Candidates</h3>
          <ul>
            {shortlisted.map((c, idx) => (
              <li key={idx}>
                <strong>{idx + 1}. {c.name}</strong><br />
                Email: {c.email}<br />
                PDF_Name: {c.pdf_name} <br />
                Match: {c.score}%<br />
                <button onClick={() => handleAccept(c)}>Accept</button>
                <button onClick={() => handleReject(c)}>Reject</button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Recruiterdashboard;

// WelcomePage.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "@fortawesome/fontawesome-free/css/all.min.css";
import { motion } from "framer-motion";
import hireImage from "../assets/img/hire3.webp";
import { register } from "../app/auth"; // <-- your auth API
import "./WelcomePage.css";

const MissionVision = () => {
  const features = [
    {
      icon: "⚡",
      title: "Streamlined Hiring",
      description: "Helpful for recruiters by saving time and making the hiring process quicker and more efficient.",
    },
    {
      icon: "💼",
      title: "Job Seeker Assistance",
      description: "Provides suggestions to job seekers for enhancing their skills and increasing employability.",
    },
    {
      icon: "🤖",
      title: "Chatbot for Interview Prep",
      description: "An AI-powered chatbot to help job seekers prepare for interviews with personalized guidance.",
    },
    {
      icon: "📄",
      title: "Resume Analyzer",
      description: "Offers detailed insights and personalized recommendations to improve your resume and stand out.",
    },
  ];

  return (
    <div className="mission-container">
      <h2 className="mission-title">Our Mission and Vision</h2>
      <div className="mission-cards">
        {features.map((feature, index) => (
          <div key={index} className="mission-card">
            <span className="mission-icon">{feature.icon}</span>
            <h3 className="mission-card-title">{feature.title}</h3>
            <p className="mission-card-description">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function WelcomePage() {
  const navigate = useNavigate();

  // State for signup form
  const [role, setRole] = useState("jobseeker");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSignup = async (e) => {
    e.preventDefault();
  
    if (!name || !email || !password || !confirmPassword) {
      alert("All fields are required");
      return;
    }
  
    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }
  
    try {
      // Call the register function from your auth API
      const res = await register(name, email, password);
      console.log("Signup response:", res);
      if (res.message === "User registered successfully") {
        alert("Signup successful! Please login.");
        navigate("/login");
      } else {
        alert(res.message || "Signup failed");
      }
    } catch (err) {
      console.error("Signup error:", err);
      alert("Error during signup");
    }
  };
  
  return (
    <div>
      {/* Navbar */}
      <header className="navbar">
        <motion.div 
          className="logo"
          initial={{ x: -100 }}
          animate={{ x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          HireSync
        </motion.div>
        <nav>
          <ul>
            <motion.li initial={{ y: -50 }} animate={{ y: 0 }} transition={{ duration: 0.8 }}>
              <a href="/">About HireSync</a>
            </motion.li>
            <motion.li initial={{ y: -50 }} animate={{ y: 0 }} transition={{ duration: 1 }}>
              <a href="/recruiter">Recruiters</a>
            </motion.li>
            <motion.li initial={{ y: -50 }} animate={{ y: 0 }} transition={{ duration: 1.2 }}>
              <a href="/JobSeeker">Job Seekers</a>
            </motion.li>
            <motion.li initial={{ y: -50 }} animate={{ y: 0 }} transition={{ duration: 1.4 }}>
              <a href="/">Contact Us</a>
            </motion.li>
          </ul>
        </nav>
      </header>

      {/* Main Section */}
      <div className="main-content">
        <div className="left-section">
          <div className="hero-content">
            <h1>
              <span className="highlight">HireSync</span>: Your Path to <strong>Smart <em>Hiring</em></strong>
            </h1>
            <p>
              Streamline your hiring process with our intuitive platform. Our resume analyzer 
              provides insights, helping candidates enhance their profiles and recruiters 
              manage applications.
            </p>
          </div>

          {/* Signup Form */}
          <div className="signup-box">
            <h2>Join <strong><em>HireSync</em></strong></h2>
            <form onSubmit={handleSignup}>
              <input
                type="text"
                placeholder="Username"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
              <input
                type="email"
                placeholder="Email Address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <input
                type="password"
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
              <button type="submit" className="signup-btn">
                Sign Up
              </button>
            </form>
            <p className="black-content">
              Already have an account? <Link to="/login">Log in</Link>
            </p>
          </div>
        </div>

        {/* Image Section */}
        <div className="right-section">
          <img className="image-struct" src={hireImage} alt="HireSync" />
        </div>
      </div>

      <MissionVision />

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div className="contact-item">
            <i className="fas fa-envelope"></i>
            <h3>Email Us</h3>
            <p>
              For any inquiries, reach us at <a href="mailto:support@example.com">support@example.com</a>
            </p>
          </div>
          <div className="contact-item">
            <i className="fas fa-phone-alt"></i>
            <h3>Call Us</h3>
            <p>
              Reach us at <a href="tel:+918309681559">+91 8309681559</a>
            </p>
          </div>
          <div className="contact-item">
            <i className="fab fa-instagram"></i>
            <h3>Instagram</h3>
            <p>
              Follow us <a href="https://instagram.com/hiresync">@HireSync</a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

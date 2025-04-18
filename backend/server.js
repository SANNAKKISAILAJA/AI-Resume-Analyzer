
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express(); // ✅ Must be defined before use

app.use(cors());
app.use(bodyParser.json());

// ✅ Import the routes AFTER app is created
const authRoutes = require("./routes");
app.use("/auth", authRoutes);

// Test endpoint
app.get("/", (_req, res) => {
  res.send("Backend is running...");
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`); // ✅ Backticks used here
});

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;
const API_KEY = process.env.API_KEY;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

// Proxy route for Gemini API
app.post('/api/generateContent', async (req, res) => {
  try {
    const { prompt } = req.body;

    const gRes = await axios.post(
      `https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key=${API_KEY}`,
      {
        contents: [{ role: 'user', parts: [{ text: prompt }] }]
      },
      {
        headers: { 'Content-Type': 'application/json' }
      }
    );

    res.json(gRes.data);
  } catch (err) {
    console.error('[Proxy error]', err.response?.data || err.message);
    res.status(500).json({ error: 'Proxy failed' });
  }
});

// Route to process video frames sent from the client
app.post('/api/analyze_face', async (req, res) => {
  try {
    // Here we would need to connect to a Python service
    // For now, let's return dummy data
    res.json({
      faces: [{
        face_id: 0,
        position: { x: 100, y: 100, width: 200, height: 200 },
        emotion: 'Happy',
        confidence: 0.95,
        eyes_closed: false,
        emotions: {
          'Angry': 0.01, 
          'Disgust': 0.01, 
          'Fear': 0.02, 
          'Happy': 0.95, 
          'Sad': 0.01, 
          'Surprise': 0.02, 
          'Neutral': 0.03
        }
      }],
      count: 1
    });
  } catch (err) {
    console.error('[Face analysis error]', err.message);
    res.status(500).json({ error: 'Face analysis failed' });
  }
});

// Start the Python Flask server
let pythonServer = null;

function startPythonServer() {
  pythonServer = spawn('python', ['face_server.py']);
  
  pythonServer.stdout.on('data', (data) => {
    console.log(`Python server: ${data}`);
  });
  
  pythonServer.stderr.on('data', (data) => {
    console.error(`Python server error: ${data}`);
  });
  
  pythonServer.on('close', (code) => {
    console.log(`Python server exited with code ${code}`);
    // Restart server if it crashes
    if (code !== 0) {
      console.log('Restarting Python server...');
      setTimeout(startPythonServer, 5000);
    }
  });
}

// Serve index.html at the root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`🔐 Server running at http://localhost:${PORT}`);
  // Start the Python server
  startPythonServer();
});

// Handle server shutdown
process.on('SIGINT', () => {
  if (pythonServer) {
    pythonServer.kill();
  }
  process.exit();
});
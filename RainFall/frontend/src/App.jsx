console.log("App loaded ✅");

import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import axios from 'axios';


function App() {
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [edaResult, setEdaResult] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:8000/upload/', formData);
      setUploadResult(res.data);
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  const runEDA = async () => {
    try {
      const res = await axios.post('http://localhost:8000/eda');
      setEdaResult(res.data);
    } catch (err) {
      console.error('EDA error:', err);
    }
  };

  const runPrediction = async () => {
    try {
      const res = await axios.post('http://localhost:8000/predict');
      setPredictionResult(res.data);
    } catch (err) {
      console.error('Prediction error:', err);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>🌦️ Rainfall Prediction App</h1>

      <h3>📁 Upload CSV File</h3>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '1rem' }}>Upload</button>
      {uploadResult && (
        <pre style={{ background: '#eee', padding: '1rem' }}>{JSON.stringify(uploadResult, null, 2)}</pre>
      )}

      <h3>🔍 Run EDA</h3>
      <button onClick={runEDA}>Run EDA</button>
      {edaResult && (
        <pre style={{ background: '#eef', padding: '1rem' }}>{JSON.stringify(edaResult, null, 2)}</pre>
      )}

      <h3>🤖 Run Prediction</h3>
      <button onClick={runPrediction}>Predict</button>
      {predictionResult && (
        <pre style={{ background: '#efe', padding: '1rem' }}>{JSON.stringify(predictionResult, null, 2)}</pre>
      )}
    </div>
  );
}

export default App;

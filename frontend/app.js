/**
 * Emotion Recognition Frontend
 * Records video from webcam, uploads to backend, displays emotion predictions
 */

const CONFIG = {
    RECORD_SECONDS: 3,
    BACKEND_URL: "http://localhost:8000",
    PREDICT_ENDPOINT: "/predict",
};

// Global state
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = null;
let timerInterval = null;

// DOM elements
const preview = document.getElementById("preview");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const uploadBtn = document.getElementById("uploadBtn");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const resultsEl = document.getElementById("results");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    initializeMediaRecorder();
    setupEventListeners();
});

/**
 * Initialize MediaRecorder with webcam stream
 */
async function initializeMediaRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240 },
            audio: true,
        });
        preview.srcObject = stream;
        statusEl.textContent = "Ready to record";
        startBtn.disabled = false;
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        console.error("Failed to get media stream:", err);
    }
}

/**
 * Setup event listeners for buttons
 */
function setupEventListeners() {
    startBtn.addEventListener("click", startRecording);
    stopBtn.addEventListener("click", stopRecording);
    uploadBtn.addEventListener("click", uploadAndPredict);
}

/**
 * Start recording video + audio
 */
function startRecording() {
    recordedChunks = [];
    const stream = preview.srcObject;
    
    if (!stream) {
        statusEl.textContent = "No media stream available";
        return;
    }

    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        handleRecordingStop();
    };

    mediaRecorder.start();
    recordingStartTime = Date.now();
    
    startBtn.disabled = true;
    stopBtn.disabled = false;
    uploadBtn.disabled = true;
    statusEl.textContent = "Recording...";
    
    startTimer();

    // Auto-stop after CONFIG.RECORD_SECONDS
    setTimeout(() => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            stopRecording();
        }
    }, CONFIG.RECORD_SECONDS * 1000);
}

/**
 * Stop recording
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
}

/**
 * Handle recording completion
 */
function handleRecordingStop() {
    clearInterval(timerInterval);
    timerEl.textContent = "";
    
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    console.log(`Recorded ${blob.size} bytes`);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    uploadBtn.disabled = false;
    statusEl.textContent = "Recording complete. Click 'Predict Emotion' to analyze.";
}

/**
 * Timer for recording duration
 */
function startTimer() {
    let elapsed = 0;
    timerInterval = setInterval(() => {
        elapsed++;
        timerEl.textContent = `${elapsed}s / ${CONFIG.RECORD_SECONDS}s`;
    }, 1000);
}

/**
 * Upload recording and get emotion predictions
 */
async function uploadAndPredict() {
    if (recordedChunks.length === 0) {
        statusEl.textContent = "No recording to upload";
        return;
    }

    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

    uploadBtn.disabled = true;
    statusEl.textContent = "Uploading and analyzing...";
    resultsEl.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing...</p></div>';

    try {
        const response = await fetch(
            `${CONFIG.BACKEND_URL}${CONFIG.PREDICT_ENDPOINT}`,
            {
                method: "POST",
                body: formData,
            }
        );

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);
        statusEl.textContent = "Analysis complete!";
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        resultsEl.innerHTML = `<p style="color: red;">Failed to get predictions: ${err.message}</p>`;
        console.error("Prediction error:", err);
    } finally {
        uploadBtn.disabled = false;
    }
}

/**
 * Display emotion prediction results
 */
function displayResults(result) {
    if (result.error) {
        resultsEl.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
        return;
    }

    const { labels, probs, top1 } = result;
    let html = "";

    // Display each emotion with progress bar
    for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        const prob = probs[i];
        const width = prob.toFixed(1);

        html += `
            <div class="result-item">
                <div class="result-label">
                    <span class="label">${label}</span>
                    <span class="prob">${width}%</span>
                </div>
                <div class="result-bar">
                    <div class="result-fill" style="width: ${width}%;">
                        ${width > 15 ? width + "%" : ""}
                    </div>
                </div>
            </div>
        `;
    }

    // Display top prediction
    html += `
        <div class="top-result">
            <h3>Top Prediction</h3>
            <div class="top-result-text">
                <strong>${top1.label}</strong> - ${top1.prob.toFixed(1)}%
            </div>
        </div>
    `;

    resultsEl.innerHTML = html;
}

// Optional: Test backend connectivity on page load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log("Backend is healthy:", data);
        }
    } catch (err) {
        console.warn("Backend not yet available:", err.message);
    }
}

checkBackendHealth();

/**
 * Emotion Recognition Frontend
 * Records video from webcam, uploads to backend, displays emotion predictions
 */

const CONFIG = {
    RECORD_SECONDS: 3,
    BACKEND_URL: resolveBackendUrl(),
    PREDICT_ENDPOINT: "/predict",
};

function resolveBackendUrl() {
    const fromWindow = typeof window !== "undefined" ? window.__EMO_BACKEND_URL : "";
    const fromQuery =
        typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("backend") : "";
    const fallback =
        typeof window !== "undefined"
            ? `${window.location.protocol}//${window.location.hostname}:8000`
            : "http://localhost:8000";
    return (fromQuery || fromWindow || fallback).replace(/\/$/, "");
}

class EmotionRecognitionApp {
    constructor(config) {
        this.config = config;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.timerInterval = null;

        this.preview = document.getElementById("preview");
        this.startBtn = document.getElementById("startBtn");
        this.stopBtn = document.getElementById("stopBtn");
        this.uploadBtn = document.getElementById("uploadBtn");
        this.statusEl = document.getElementById("status");
        this.timerEl = document.getElementById("timer");
        this.resultsEl = document.getElementById("results");
    }

    async initialize() {
        await this.initializeMediaRecorder();
        this.setupEventListeners();
        await this.checkBackendHealth();
    }

    async initializeMediaRecorder() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240 },
                audio: true,
            });
            this.preview.srcObject = stream;
            this.statusEl.textContent = "Ready to record";
            this.startBtn.disabled = false;
        } catch (err) {
            this.statusEl.textContent = `Error: ${err.message}`;
            console.error("Failed to get media stream:", err);
        }
    }

    setupEventListeners() {
        this.startBtn.addEventListener("click", () => this.startRecording());
        this.stopBtn.addEventListener("click", () => this.stopRecording());
        this.uploadBtn.addEventListener("click", () => this.uploadAndPredict());
    }

    startRecording() {
        this.recordedChunks = [];
        const stream = this.preview.srcObject;
        if (!stream) {
            this.statusEl.textContent = "No media stream available";
            return;
        }

        this.mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };
        this.mediaRecorder.onstop = () => this.handleRecordingStop();
        this.mediaRecorder.start();

        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.uploadBtn.disabled = true;
        this.statusEl.textContent = "Recording...";
        this.startTimer();

        setTimeout(() => {
            if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
                this.stopRecording();
            }
        }, this.config.RECORD_SECONDS * 1000);
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
            this.mediaRecorder.stop();
        }
    }

    handleRecordingStop() {
        clearInterval(this.timerInterval);
        this.timerEl.textContent = "";
        const blob = new Blob(this.recordedChunks, { type: "video/webm" });
        console.log(`Recorded ${blob.size} bytes`);
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.uploadBtn.disabled = false;
        this.statusEl.textContent = "Recording complete. Click 'Predict Emotion' to analyze.";
    }

    startTimer() {
        let elapsed = 0;
        this.timerInterval = setInterval(() => {
            elapsed++;
            this.timerEl.textContent = `${elapsed}s / ${this.config.RECORD_SECONDS}s`;
        }, 1000);
    }

    async uploadAndPredict() {
        if (this.recordedChunks.length === 0) {
            this.statusEl.textContent = "No recording to upload";
            return;
        }

        const blob = new Blob(this.recordedChunks, { type: "video/webm" });
        const formData = new FormData();
        formData.append("file", blob, "recording.webm");

        this.uploadBtn.disabled = true;
        this.statusEl.textContent = "Uploading and analyzing...";
        this.resultsEl.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing...</p></div>';

        try {
            const response = await fetch(`${this.config.BACKEND_URL}${this.config.PREDICT_ENDPOINT}`, {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            const result = await response.json();
            this.displayResults(result);
            this.statusEl.textContent = "Analysis complete!";
        } catch (err) {
            this.statusEl.textContent = `Error: ${err.message}`;
            this.resultsEl.innerHTML = `<p style="color: red;">Failed to get predictions: ${err.message}</p>`;
            console.error("Prediction error:", err);
        } finally {
            this.uploadBtn.disabled = false;
        }
    }

    displayResults(result) {
        if (result.error) {
            this.resultsEl.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
            return;
        }

        const { labels, probs, top1 } = result;
        let html = "";
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

        html += `
            <div class="top-result">
                <h3>Top Prediction</h3>
                <div class="top-result-text">
                    <strong>${top1.label}</strong> - ${top1.prob.toFixed(1)}%
                </div>
            </div>
        `;
        this.resultsEl.innerHTML = html;
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.config.BACKEND_URL}/health`);
            if (response.ok) {
                const data = await response.json();
                console.log("Backend is healthy:", data);
            }
        } catch (err) {
            console.warn("Backend not yet available:", err.message);
        }
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    const app = new EmotionRecognitionApp(CONFIG);
    await app.initialize();
});

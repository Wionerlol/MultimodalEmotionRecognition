/**
 * Emotion Recognition Frontend
 * Supports real-time streaming over WebSocket and legacy one-shot prediction.
 */

const CONFIG = {
    BACKEND_URL: resolveBackendUrl(),
    PREDICT_ENDPOINT: "/predict",
    STREAM_ENDPOINT: "/ws/stream",
    STREAM_FRAME_INTERVAL_MS: 250,
    AUDIO_BUFFER_SIZE: 2048,
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

function toWebSocketUrl(httpUrl, endpoint) {
    const url = new URL(endpoint, httpUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
}

function float32ToInt16(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const sample = Math.max(-1, Math.min(1, float32Array[i]));
        int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return int16;
}

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

class EmotionRecognitionApp {
    constructor(config) {
        this.config = config;
        this.stream = null;
        this.websocket = null;
        this.streaming = false;
        this.streamStartTime = 0;
        this.frameInterval = null;
        this.audioContext = null;
        this.audioSource = null;
        this.audioProcessor = null;
        this.canvas = document.createElement("canvas");
        this.lastPredictionTs = 0;

        this.preview = document.getElementById("preview");
        this.startBtn = document.getElementById("startBtn");
        this.stopBtn = document.getElementById("stopBtn");
        this.uploadBtn = document.getElementById("uploadBtn");
        this.statusEl = document.getElementById("status");
        this.timerEl = document.getElementById("timer");
        this.resultsEl = document.getElementById("results");
    }

    async initialize() {
        await this.initializeMediaStream();
        this.setupEventListeners();
        await this.checkBackendHealth();
    }

    async initializeMediaStream() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240 },
                audio: true,
            });
            this.preview.srcObject = this.stream;
            this.statusEl.textContent = "Ready for live streaming";
            this.startBtn.disabled = false;
        } catch (err) {
            this.statusEl.textContent = `Error: ${err.message}`;
            console.error("Failed to get media stream:", err);
        }
    }

    setupEventListeners() {
        this.startBtn.addEventListener("click", () => this.startStreaming());
        this.stopBtn.addEventListener("click", () => this.stopStreaming());
        this.uploadBtn.addEventListener("click", () => this.runOneShotPrediction());
    }

    async startStreaming() {
        if (!this.stream || this.streaming) {
            return;
        }

        const wsUrl = toWebSocketUrl(this.config.BACKEND_URL, this.config.STREAM_ENDPOINT);
        this.websocket = new WebSocket(wsUrl);
        this.statusEl.textContent = "Connecting to streaming backend...";

        this.websocket.onopen = async () => {
            this.streaming = true;
            this.streamStartTime = performance.now();
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.uploadBtn.disabled = true;
            this.resultsEl.innerHTML = '<p class="placeholder">Streaming started. Waiting for the first prediction...</p>';
            this.websocket.send(JSON.stringify({ type: "start", timestamp: Date.now() / 1000 }));
            this.startTimer();
            this.startFrameStreaming();
            await this.startAudioStreaming();
            this.statusEl.textContent = "Streaming live. Backend is running sliding-window inference...";
        };

        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === "prediction") {
                this.lastPredictionTs = Date.now();
                this.displayResults(message.payload, true);
                this.statusEl.textContent = "Live prediction updated";
            } else if (message.type === "error") {
                this.statusEl.textContent = `Streaming error: ${message.detail}`;
            }
        };

        this.websocket.onerror = () => {
            this.statusEl.textContent = "Streaming connection error";
        };

        this.websocket.onclose = () => {
            this.cleanupStreamingState();
        };
    }

    stopStreaming() {
        if (!this.streaming) {
            return;
        }
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: "flush", timestamp: Date.now() / 1000 }));
            this.websocket.send(JSON.stringify({ type: "stop", timestamp: Date.now() / 1000 }));
            this.websocket.close();
        } else {
            this.cleanupStreamingState();
        }
    }

    cleanupStreamingState() {
        this.streaming = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.uploadBtn.disabled = false;
        clearInterval(this.frameInterval);
        this.frameInterval = null;
        this.stopTimer();

        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor.onaudioprocess = null;
            this.audioProcessor = null;
        }
        if (this.audioSource) {
            this.audioSource.disconnect();
            this.audioSource = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        this.websocket = null;
        this.statusEl.textContent = "Streaming stopped";
    }

    startFrameStreaming() {
        const width = this.preview.videoWidth || 320;
        const height = this.preview.videoHeight || 240;
        this.canvas.width = width;
        this.canvas.height = height;
        const ctx = this.canvas.getContext("2d");

        this.frameInterval = setInterval(() => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            ctx.drawImage(this.preview, 0, 0, width, height);
            const imageB64 = this.canvas.toDataURL("image/jpeg", 0.7);
            this.websocket.send(
                JSON.stringify({
                    type: "frame",
                    timestamp: Date.now() / 1000,
                    image_b64: imageB64,
                }),
            );
        }, this.config.STREAM_FRAME_INTERVAL_MS);
    }

    async startAudioStreaming() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.audioSource = this.audioContext.createMediaStreamSource(this.stream);
        this.audioProcessor = this.audioContext.createScriptProcessor(this.config.AUDIO_BUFFER_SIZE, 1, 1);

        this.audioProcessor.onaudioprocess = (event) => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            const float32 = event.inputBuffer.getChannelData(0);
            const pcm16 = float32ToInt16(float32);
            this.websocket.send(
                JSON.stringify({
                    type: "audio",
                    timestamp: Date.now() / 1000,
                    sample_rate: this.audioContext.sampleRate,
                    pcm_b64: arrayBufferToBase64(pcm16.buffer),
                }),
            );
        };

        this.audioSource.connect(this.audioProcessor);
        this.audioProcessor.connect(this.audioContext.destination);
    }

    async runOneShotPrediction() {
        if (!this.stream) {
            this.statusEl.textContent = "No media stream available";
            return;
        }

        this.statusEl.textContent = "Recording a 3-second clip for one-shot prediction...";
        this.resultsEl.innerHTML = '<div class="loading"><div class="spinner"></div><p>Recording and processing...</p></div>';

        try {
            const recorder = new MediaRecorder(this.stream, { mimeType: "video/webm" });
            const chunks = [];
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };

            const done = new Promise((resolve) => {
                recorder.onstop = resolve;
            });

            recorder.start();
            await new Promise((resolve) => setTimeout(resolve, 3000));
            recorder.stop();
            await done;

            const blob = new Blob(chunks, { type: "video/webm" });
            const formData = new FormData();
            formData.append("file", blob, "recording.webm");

            const response = await fetch(`${this.config.BACKEND_URL}${this.config.PREDICT_ENDPOINT}`, {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            const result = await response.json();
            this.displayResults(result, false);
            this.statusEl.textContent = "One-shot prediction complete";
        } catch (err) {
            this.statusEl.textContent = `Error: ${err.message}`;
            this.resultsEl.innerHTML = `<p style="color: red;">Failed to get predictions: ${err.message}</p>`;
            console.error("Prediction error:", err);
        }
    }

    startTimer() {
        this.stopTimer();
        this.timerEl.textContent = "0.0s live";
        this.timerInterval = setInterval(() => {
            const elapsed = (performance.now() - this.streamStartTime) / 1000;
            const sincePrediction = this.lastPredictionTs ? ` | last update ${(Date.now() - this.lastPredictionTs) / 1000}s ago` : "";
            this.timerEl.textContent = `${elapsed.toFixed(1)}s live${sincePrediction}`;
        }, 200);
    }

    stopTimer() {
        clearInterval(this.timerInterval);
        this.timerEl.textContent = "";
    }

    displayResults(result, liveMode) {
        if (result.error) {
            this.resultsEl.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
            return;
        }

        const { labels, probs, top1 } = result;
        let html = liveMode
            ? `<div class="top-result"><h3>Live Prediction</h3><div class="top-result-text"><strong>${top1.label}</strong> - ${top1.prob.toFixed(1)}%</div></div>`
            : "";

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
                <h3>${liveMode ? "Current Top Prediction" : "Top Prediction"}</h3>
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

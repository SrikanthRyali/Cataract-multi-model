# Cataract Hub: Desktop Application Overview

## 1. Application Overall Workflow (Desktop ONLY)
The Cataract Hub Desktop application provides a seamless, high-performance experience for eye screening without the need for a web browser. The workflow is designed for speed and clinical accuracy:

1.  **Launch**: The user opens the Cataract Hub application. The Electron shell initializes a native window and loads the local `index.html`.
2.  **Configuration**: At first launch, the user defines their **Groq API Key** in the settings drawer. This key is saved locally in the browser's `localStorage` for persistence.
3.  **Image Upload**: The user selects or drags-and-drops a close-up image of a single eye. The application provides photographic guides (well-lit, steady camera) to ensure optimal results.
4.  **Neural Analysis**: When "Start Vision Analysis" is clicked, the application converts the image to a Base64 string and sends it via an **IPC (Inter-Process Communication)** bridge to the Main process.
5.  **Cloud Inference**: The Main process connects to the **Hugging Face Gradio Space** (`Srikanth22MH1A42C6/model-api-2`). It runs five specialized CNN models (DeepCNN, ResNet, VGG, AlexNet, DeepANN) in parallel.
6.  **Result Aggregation**: The "Ensemble" system collects votes from all five models. A majority vote determines the final diagnosis (Cataract vs. Normal).
7.  **Diagnostic Display**: The UI updates instantly with count-up animations for confidence scores, a 4-tier clinical metric grid, and a 3-tier detailed explanation (Simple, Technical, and AI-Model logic).
8.  **AI Report Generation**: Simultaneously, the results are sent to the **Groq Llama-3.3 70B** model to generate a structured, professional medical screening report.
9.  **Interactive Support**: The user can then interact with the "Medical AI" chatbot at the bottom-right to ask follow-up questions in English, Telugu, or Hindi.

---

## 2. System Components (Start to End)
The system is built on a "Three-Tier Architecture" adapted for desktop use:

*   **Tier 1: Desktop Shell (Electron)**
    *   Acts as the container. It manages the operating system window, native menus, and the secure bridge between the web content and the system resources.
*   **Tier 2: Logic & Bridge (Renderer + Preload)**
    *   The **Renderer** handles the UI (HTML/CSS) and user interactions.
    *   The **Preload Script** acts as a secure "guard," exposing only specific APIs (like Gradio and Groq calls) to the UI while preventing malicious access to the machine.
*   **Tier 3: Cloud Intelligence (HF + Groq)**
    *   **Hugging Face Spaces**: Hosts the heavy Deep Learning models. This allows the desktop app to remain lightweight while still performing massive calculations in the cloud.
    *   **Groq API**: Provides ultra-fast LLM (Large Language Model) processing for the AI Report and Chatbot.

---

## 3. Frontend & Backend Flow Detail
### Frontend Flow (The "Visible" App)
- **UI Structure**: Built with **Tailwind CSS**, utilizing a "Glassmorphism" aesthetic for a premium medical feel.
- **State Management**: Uses pure JavaScript to track the current image, diagnosis, and chat history.
- **Animations**: Uses CSS Transitions and a custom `animateValue` function for "count-up" effects on percentages.

### Backend Flow (The "Invisible" Processing)
- **Inference Pipeline**: The Backend (on Hugging Face) takes the image, validates it (7-layer check), and passes it to the ensemble.
- **LLM Pipeline**: The AI Report uses a specialized prompt that treats the diagnostic data as a "Knowledge Base," ensuring the AI doesn't hallucinate.

---

## 4. Connection Logic (IPC & Context Bridge)
In Electron, the Frontend (Renderer) and Backend (Main Process) are strictly separated for security. We connect them using **IPC (Inter-Process Communication)**:

1.  **Context Bridge**: In `preload.js`, we use `contextBridge.exposeInMainWorld('api', ...)` to create a secure port called `window.api`.
2.  **IPC Renderer**: When the UI needs data (like a prediction), it calls `ipcRenderer.invoke('gradio-call', data)`.
3.  **IPC Main**: In `main.js`, the app "listens" for these requests using `ipcMain.handle`. It performs the Gradio connection and returns the result back to the UI.

---

## 5. Deployment Guide (How & Where)
### Current Environment: Development
- **Execution**: Run `npm start` in the `desktop-application` folder. This launches the app using the local Electron binary.
- **Prerequisites**: Requires Node.js (v18+) and an active internet connection.

### Future Environment: Final Distribution
- **Packaging**: Use `electron-builder` or `electron-forge` to compile the app into a standalone file.
- **Windows**: Produces a `.exe` setup file or a portable executable.
- **macOS/Linux**: Produces `.dmg`, `.app`, or `.deb` files.
- **Deployment Hub**: The final binary can be hosted on GitHub Releases, a project website, or distributed via local storage (USB).

---

## 6. Core Functions & Logic
### Ensemble Voting Logic
Instead of relying on one model, we use five:
1.  **DeepCNN**: Specialized in broad feature extraction.
2.  **ResNet**: Deep residual learning for intricate textures.
3.  **VGG**: Simpler, consistent pattern recognition.
4.  **AlexNet**: Fast, baseline architectural check.
5.  **DeepANN**: Analyzes flattened intensity histograms.
*Majority Rule: If 3 or more models detect Cataract, the final verdict is Cataract.*

### AI Report Generator
Uses **Llama-3.3 70B** with a clinical prompt. It takes the model agreement (count) and confidence percentage to draft a structured report covering "What is Cataract," "Causes," and "Food to Eat."

---

## 7. Tech Stack Usage
- **Electron**: The foundational shell for the desktop experience.
- **Tailwind CSS**: Rapid, modern styling for a premium UI.
- **JavaScript (ES6+)**: The entire logic layer from UI reactivity to API calls.
- **Markdown-it**: Used to render the AI Report from raw text into beautiful HTML.
- **@gradio/client**: The library used in the Main process to communicate with Hugging Face.
- **Groq SDK (Fetch-based)**: High-speed connection to the Llama models.

---

## 8. File-by-File Breakdown (`desktop-application` directory)
1.  **`index.html`**:
    - **Purpose**: The entire layout.
    - **Logic**: Contains all UI components (Navbar, Upload Zone, Results Grid, AI Report Section, Settings Drawer, and Chat Window).
2.  **`renderer.js`**:
    - **Purpose**: The "Brain" of the Frontend.
    - **Logic**: Handles button clicks, image previews, UI state changes, triggering the analysis, and updating results on the screen.
3.  **`main.js`**:
    - **Purpose**: The "Heart" of the App.
    - **Logic**: Manages the native window, initializes the Gradio client, and handles the `ipcMain` calls for inference and chat.
4.  **`preload.js`**:
    - **Purpose**: The Secure "Bridge".
    - **Logic**: Exposes the `window.api` functions. It contains the logic for the Groq Summarizer and the Chat assistant using `fetch` calls.
5.  **`package.json`**:
    - **Purpose**: The "Identity" file.
    - **Logic**: Lists all dependencies (`electron`, `@gradio/client`, `axios`) and the startup scripts.

---

## 9. Software & Hardware Requirements
### Software Requirements
- **OS**: Windows 10/11 (64-bit), macOS 10.15+, or modern Linux (Ubuntu/Fedora).
- **Runtime**: Node.js v18 or later (for development).
- **API Keys**: A valid Groq Cloud API Key.

### Hardware Requirements
- **CPU**: Quad-core 2.0GHz or better.
- **RAM**: 4GB Minimum (8GB Recommended).
- **Storage**: 200MB for the application, plus cache.
- **Display**: 1280x800 resolution or higher.
- **Network**: Broadband connection (for cloud-based AI inference).

---

## 10. Physical Environment Requirements
For accurate screening, the user must ensure:
- **Lighting**: Bright, even lighting. Avoid glare on the eye surface.
- **Distance**: The camera should be 10-15cm from the eye.
- **Stability**: Handheld phones should be rested on a surface or held steady.
- **Focus**: The pupil must be clearly visible and in sharp focus.

---

## 11. Resources & Security Requirements
### Resource Management
- **Memory**: Electron creates multiple processes. The app is optimized to keep background usage low when idle.
- **Network**: Each analysis consumes ~500KB of data (image upload + text result).

### Security Requirements
- **API Privacy**: API keys are stored only in the user's local profile (`localStorage`). They are never sent to any server other than Groq.
- **Image Privacy**: Images are processed in RAM and sent over HTTPS. They are not permanently stored on the cloud servers.
- **Execution Security**: `contextIsolation` is enabled to prevent XSS attacks from reaching the local file system.

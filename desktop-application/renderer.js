const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const uploadInstruction = document.getElementById('upload-instruction');
const predictBtn = document.getElementById('predict-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStatus = document.getElementById('loading-status');
const resultsContainer = document.getElementById('results-container');
const verdictCard = document.getElementById('verdict-card');
const verdictIcon = document.getElementById('verdict-icon');
const verdictText = document.getElementById('verdict-text');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar = document.getElementById('confidence-bar');
const modelList = document.getElementById('model-list');
const aiReportContent = document.getElementById('ai-report-content');
const btnSettings = document.getElementById('btn-settings');
const subpageOverlay = document.getElementById('subpage-overlay');
const btnCloseSettings = document.getElementById('btn-close-settings');
const saveSettings = document.getElementById('save-settings');
const groqApiKeyInput = document.getElementById('groq-api-key');
const toast = document.getElementById('result-toast');

let currentImageBase64 = null;
let groqApiKey = '';
const DEFAULT_SPACE_ID = 'Srikanth22MH1A42C6/model-api';

// UI Interactions
btnSettings.onclick = () => subpageOverlay.style.display = 'flex';
btnCloseSettings.onclick = () => subpageOverlay.style.display = 'none';

saveSettings.onclick = () => {
    groqApiKey = groqApiKeyInput.value;
    subpageOverlay.style.display = 'none';
};

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            currentImageBase64 = event.target.result;
            imagePreview.src = currentImageBase64;
            previewContainer.classList.remove('hidden');
            uploadInstruction.classList.add('hidden');
            predictBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
};

predictBtn.onclick = async () => {
    if (!currentImageBase64) return;
    if (!groqApiKey) {
        alert("Please set your Groq API Key in Settings first.");
        subpageOverlay.style.display = 'flex';
        return;
    }

    loadingOverlay.style.display = 'flex';
    resultsContainer.style.display = 'none';
    
    try {
        loadingStatus.innerText = "Connecting to Neural Ensemble Space...";
        
        // Calling the Gradio Space API
        // Gradio predict [image, groq_key] -> returns result, votes, individual_str, summary, simple, technical, heatmap
        const resultData = await window.api.huggingface.predict(DEFAULT_SPACE_ID, currentImageBase64, groqApiKey);
        
        // Parsing Gradio results
        // 0: Final Pred String
        // 1: Votes String
        // 2: Individual Results String
        // 3: AI Summary
        // 4: Simple Explanation
        // 5: Technical Explanation
        // 6: Heatmap source

        const predMatch = resultData[0].match(/Prediction:\s*(Cataract|Normal)\s*\(([\d.]+)%\)/i);
        const finalPred = predMatch ? predMatch[1] : 'Normal';
        const confidence = predMatch ? predMatch[2] : '0';
        const aiSummary = resultData[3];
        const individualStr = resultData[2];

        // Format individual results for the UI
        const individualResults = individualStr.split('\n').filter(l => l.includes(':')).map(line => {
            const parts = line.split(':');
            const modelName = parts[0].trim();
            const predParts = parts[1].match(/(Cataract|Normal)\s*\(([\d.]+)%\)/i);
            return {
                model: modelName,
                prediction: predParts ? predParts[1] : 'Normal',
                confidence: predParts ? predParts[2] : '0'
            };
        });

        updateUI(finalPred, confidence, individualResults);

        // Update AI Report
        const md = window.markdownit();
        aiReportContent.innerHTML = md.render(aiSummary || "No report generated.");

        loadingOverlay.style.display = 'none';
        resultsContainer.style.display = 'block';
        showToast(finalPred, confidence);
        
    } catch (err) {
        if (err.message && err.message.includes('timeout')) {
            alert("Connection error: The Hugging Face Space might be sleeping. I've sent a 'wake up' signal. Please try again in 10-20 seconds.");
        } else {
            console.error("Analysis Error:", err);
            alert("An error occurred during analysis: " + (err.response?.data?.error || err.message));
        }
        loadingOverlay.style.display = 'none';
    }
};

function updateUI(prediction, confidence, individualResults) {
    verdictText.innerText = prediction;
    confidenceText.innerText = `${confidence}%`;
    confidenceBar.style.width = `${confidence}%`;
    
    if (prediction === 'Cataract') {
        verdictIcon.innerText = '🚨';
        verdictCard.className = 'glass rounded-3xl p-8 lg:p-12 flex flex-col items-center justify-center text-center shadow-xl card-lift border-2 border-rose-200 bg-rose-50/30';
        verdictText.className = 'font-display font-extrabold mb-2 text-rose-600';
        confidenceBar.className = 'h-full bg-rose-500';
    } else {
        verdictIcon.innerText = '✨';
        verdictCard.className = 'glass rounded-3xl p-8 lg:p-12 flex flex-col items-center justify-center text-center shadow-xl card-lift border-2 border-teal-200 bg-teal-50/30';
        verdictText.className = 'font-display font-extrabold mb-2 text-teal-600';
        confidenceBar.className = 'h-full bg-teal-500';
    }

    modelList.innerHTML = individualResults.map(res => `
        <div class="bg-white/70 px-5 py-4 rounded-2xl flex justify-between items-center border border-slate-50 hover:shadow-md transition-all">
            <div class="flex items-center gap-3 min-w-0">
                <div class="w-3 h-3 rounded-full shrink-0 ${res.prediction === 'Cataract' ? 'bg-rose-400' : 'bg-teal-400'}"></div>
                <span class="font-semibold text-slate-700 truncate">${res.model}</span>
            </div>
            <div class="text-right shrink-0 ml-4">
                <div class="font-bold text-slate-400 text-sm">${res.confidence}%</div>
                <div class="font-bold text-sm ${res.prediction === 'Cataract' ? 'text-rose-500' : 'text-teal-600'}">${res.prediction}</div>
            </div>
        </div>
    `).join('');
}

function showToast(prediction, confidence) {
    const indicator = document.getElementById('toast-indicator');
    const text = document.getElementById('toast-text');
    
    indicator.className = `toast-dot ${prediction === 'Cataract' ? 'bg-rose-400' : 'bg-teal-400'}`;
    text.innerText = `Analysis Complete — ${prediction} (${confidence}%)`;
    
    toast.style.display = 'flex';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 4000);
}

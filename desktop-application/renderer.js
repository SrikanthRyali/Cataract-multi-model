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
const hfTokenInput = document.getElementById('hf-token');
const toast = document.getElementById('result-toast');

let currentImageBase64 = null;
let groqApiKey = '';
let hfToken = '';

const MODELS = [
    { name: 'ResNet', id: 'Srikanth22MH1A42C6/cataract-classification-resnet' },
    { name: 'VGG-16', id: 'Srikanth22MH1A42C6/cataract-classification-vgg' },
    { name: 'AlexNet', id: 'Srikanth22MH1A42C6/cataract-classification-alexnet' },
    { name: 'DeepCNN', id: 'Srikanth22MH1A42C6/cataract-classification-deepcnn' },
    { name: 'DeepANN', id: 'Srikanth22MH1A42C6/cataract-classification-deepann' }
];

// UI Interactions
btnSettings.onclick = () => subpageOverlay.style.display = 'flex';
btnCloseSettings.onclick = () => subpageOverlay.style.display = 'none';

saveSettings.onclick = () => {
    groqApiKey = groqApiKeyInput.value;
    hfToken = hfTokenInput.value;
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
    if (!groqApiKey || !hfToken) {
        alert("Please set Groq and Hugging Face keys in Settings first.");
        subpageOverlay.style.display = 'flex';
        return;
    }

    loadingOverlay.style.display = 'flex';
    resultsContainer.style.display = 'none';
    
    try {
        const imageData = currentImageBase64.split(',')[1];
        const binaryData = Uint8Array.from(atob(imageData), c => c.charCodeAt(0));

        let modelResults = [];
        let cataractVotes = 0;
        let normalVotes = 0;

        for (const model of MODELS) {
            loadingStatus.innerText = `Consulting ${model.name}...`;
            try {
                // Call Hf Inference API
                const result = await window.api.huggingface.inference(model.id, null, binaryData, hfToken);
                
                // Assuming result format: [{ label: 'Cataract', score: 0.9 }, { label: 'Normal', score: 0.1 }]
                const topLabel = result[0].label;
                const topScore = (result[0].score * 100).toFixed(2);
                
                modelResults.push({ model: model.name, prediction: topLabel, confidence: topScore });
                if (topLabel === 'Cataract') cataractVotes++;
                else normalVotes++;
            } catch (err) {
                console.error(`Error with model ${model.name}:`, err);
                modelResults.push({ model: model.name, prediction: 'Error', confidence: 0 });
            }
        }

        const finalPred = cataractVotes >= normalVotes ? 'Cataract' : 'Normal';
        const avgConfidence = (modelResults
            .filter(m => m.prediction === finalPred)
            .reduce((acc, curr) => acc + parseFloat(curr.confidence), 0) / (finalPred === 'Cataract' ? cataractVotes : normalVotes) || 0).toFixed(2);

        updateUI(finalPred, avgConfidence, modelResults);

        // Call Groq for Summary
        loadingStatus.innerText = "Generating AI Clinical Report...";
        const findings = `Diagnosis: ${finalPred}, Confidence: ${avgConfidence}%, Model Support: ${cataractVotes}/5 models agreed.`;
        const summary = await window.api.groq.summarize(findings, groqApiKey);
        
        const md = window.markdownit();
        aiReportContent.innerHTML = md.render(summary);

        loadingOverlay.style.display = 'none';
        resultsContainer.style.display = 'block';
        showToast(finalPred, avgConfidence);
        
    } catch (err) {
        console.error("Analysis Error:", err);
        alert("An error occurred during analysis. Check console for details.");
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

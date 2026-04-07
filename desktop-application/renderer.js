// ─────────────────────────────────────────────
//  Cataract Hub — renderer.js
//  All UI logic for the Electron renderer process
// ─────────────────────────────────────────────

'use strict';

// ── Constants ──────────────────────────────────
const SPACE_ID   = 'Srikanth22MH1A42C6/model-api-2';
const API_NAME   = '/predict_ensemble';
const md         = window.markdownit ? window.markdownit() : null;

// ── State ──────────────────────────────────────
let selectedFile     = null;   // File object
let selectedDataURL  = null;   // base64 data-url of the eye image
let lastDiagnostic   = null;   // result object from last prediction
let chatHistory      = [];     // sliding window [{role,content}]
const MAX_CHAT_HIST  = 10;

// ── DOM refs ───────────────────────────────────
const dropZone           = document.getElementById('drop-zone');
const fileInput          = document.getElementById('file-input');
const imagePreviewCont   = document.getElementById('image-preview-container');
const imagePreview       = document.getElementById('image-preview');
const uploadInstruction  = document.getElementById('upload-instruction');
const predictBtn         = document.getElementById('predict-btn');
const resultsRoot        = document.getElementById('results-root');
const loadingOverlay     = document.getElementById('loading-overlay');
const resultToast        = document.getElementById('result-toast');
const toastText          = document.getElementById('toast-text');
const subpageOverlay     = document.getElementById('subpage-overlay');
const chatPanel          = document.getElementById('chat-panel');
const chatMessages       = document.getElementById('chat-messages');
const chatForm           = document.getElementById('chat-form');
const chatInput          = document.getElementById('chat-input');

// ── Settings (localStorage) ────────────────────
function loadSettings() {
  const groqKey = localStorage.getItem('groq_api_key') || '';
  const hfToken = localStorage.getItem('hf_token') || '';
  const groqInput = document.getElementById('groq-api-key-input');
  const hfInput   = document.getElementById('hf-token-input');
  if (groqInput) groqInput.value = groqKey;
  if (hfInput)   hfInput.value   = hfToken;
}

function saveSettings() {
  const groqKey = document.getElementById('groq-api-key-input').value.trim();
  const hfToken = document.getElementById('hf-token-input').value.trim();
  localStorage.setItem('groq_api_key', groqKey);
  localStorage.setItem('hf_token', hfToken);
  showToast('✓ Settings saved', '#10b981');
  closeSubpage();
}

function getGroqKey()  { return localStorage.getItem('groq_api_key') || ''; }
function getHfToken()  { return localStorage.getItem('hf_token')     || ''; }

// ── Image upload / preview ─────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('⚠ Please upload a valid image file', '#ef4444');
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    selectedDataURL = e.target.result;

    // Show preview, hide placeholder text
    imagePreview.src = selectedDataURL;
    imagePreviewCont.style.display = 'flex';
    uploadInstruction.style.display = 'none';

    // Show predict button
    predictBtn.classList.remove('hidden');
    predictBtn.disabled = false;

    // Reset previous results
    resultsRoot.style.display = 'none';
    lastDiagnostic = null;
  };
  reader.readAsDataURL(file);
}

// Click on drop-zone triggers the hidden input
dropZone.addEventListener('click', (e) => {
  // Don't re-trigger if clicking the input itself
  if (e.target !== fileInput) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    handleFile(fileInput.files[0]);
  }
});

// Drag-and-drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ── Predict ────────────────────────────────────
predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!selectedDataURL) {
    showToast('⚠ Please upload an eye image first', '#ef4444');
    return;
  }

  const groqKey = getGroqKey();
  if (!groqKey) {
    showToast('⚠ Please add your Groq API key in Config', '#f59e0b');
    openSubpage('config');
    return;
  }

  setLoading(true);

  try {
    // ── Call Gradio backend ──
    const payload = {
      image:        selectedDataURL,
      groq_api_key: groqKey,
    };
    const hfToken = getHfToken();
    if (hfToken) payload.hfToken = hfToken;

    const data = await window.api.huggingface.call(SPACE_ID, API_NAME, payload);

    // data[0] is expected to be a JSON string or object from the Gradio API
    let result = data[0];
    if (typeof result === 'string') {
      try { result = JSON.parse(result); } catch (_) { /* raw string */ }
    }

    lastDiagnostic = result;
    renderResults(result, groqKey);
  } catch (err) {
    console.error('Prediction error:', err);
    showToast('✕ Analysis failed — check your API keys or connection', '#ef4444');
  } finally {
    setLoading(false);
  }
}

// ── Render results ─────────────────────────────
function renderResults(data, groqKey) {
  if (!data) return;

  // ─ Verdict card ─
  const isCataract = String(data.verdict || data.prediction || '').toLowerCase().includes('cataract');
  const confidence  = parseFloat(data.confidence ?? data.avg_confidence ?? 0) * (data.confidence > 1 ? 1 : 100);
  const confPct     = Math.round(confidence > 1 ? confidence : confidence * 100);

  document.getElementById('verdict-text').textContent = isCataract ? 'Cataract' : 'Normal';
  document.getElementById('confidence-text').textContent = `${confPct}%`;
  const bar = document.getElementById('confidence-bar');
  bar.style.width = '0%';
  bar.style.background = isCataract ? '#e11d48' : '#0d9488';
  setTimeout(() => { bar.style.width = confPct + '%'; }, 50);

  const iconCont = document.getElementById('verdict-icon-container');
  iconCont.textContent = isCataract ? '⚠️' : '✅';
  iconCont.style.background = isCataract
    ? 'linear-gradient(135deg,rgba(225,29,72,.15),rgba(225,29,72,.05))'
    : 'linear-gradient(135deg,rgba(13,148,136,.15),rgba(13,148,136,.05))';
  iconCont.className = `rounded-3xl flex items-center justify-center mb-5 shadow-lg ${isCataract ? 'verdict-glow-cataract' : 'verdict-glow-normal'}`;

  document.getElementById('verdict-disclaimer').classList.toggle('hidden', !isCataract);

  // ─ Feature tags ─
  const tagsEl = document.getElementById('feature-tags');
  tagsEl.innerHTML = '';
  const tags = data.tags || data.feature_tags || [];
  tags.forEach(t => {
    const span = document.createElement('span');
    span.className = `feat-tag ${isCataract ? 'feat-tag-warn' : 'feat-tag-ok'}`;
    span.textContent = t;
    tagsEl.appendChild(span);
  });

  // ─ Scanned image ─
  document.getElementById('scanned-image').src = selectedDataURL;
  document.getElementById('image-overlay-verdict').textContent = isCataract ? '⚠ Cataract Detected' : '✓ Normal';
  const overlayConf = document.getElementById('image-overlay-conf');
  overlayConf.textContent = `${confPct}% confidence`;
  overlayConf.style.background = isCataract ? '#e11d48' : '#0d9488';

  // ─ Model intelligence list ─
  const modelList = document.getElementById('model-intelligence-list');
  modelList.innerHTML = '';
  const models = data.models || data.model_votes || [];
  models.forEach(m => {
    const isCat = String(m.prediction || m.result || '').toLowerCase().includes('cataract');
    const mConf  = Math.round((parseFloat(m.confidence ?? 0)) * (m.confidence > 1 ? 1 : 100));
    const div    = document.createElement('div');
    div.className = 'flex items-center gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100';
    div.innerHTML = `
      <div class="w-2 h-2 rounded-full shrink-0" style="background:${isCat ? '#e11d48' : '#0d9488'}"></div>
      <div class="flex-1 min-w-0">
        <p class="text-xs font-bold text-slate-800 truncate">${m.name || m.model || 'Model'}</p>
        <div class="feat-bar-track mt-1"><div class="feat-bar-fill" style="width:${mConf}%;background:${isCat ? '#e11d48' : '#0d9488'}"></div></div>
      </div>
      <span class="text-[10px] font-extrabold shrink-0" style="color:${isCat ? '#e11d48' : '#0d9488'}">${mConf}%</span>
    `;
    modelList.appendChild(div);
  });

  // ─ Metrics grid ─
  const metricsGrid = document.getElementById('metrics-grid');
  metricsGrid.innerHTML = '';
  const metrics = data.metrics || data.clinical_features || [];
  const colors  = ['rose', 'teal', 'amber', 'violet'];
  metrics.forEach((metric, i) => {
    const c      = colors[i % colors.length];
    const pct    = Math.round(parseFloat(metric.value ?? metric.score ?? 0) * 100);
    const card   = document.createElement('div');
    card.className = `feat-card feat-card-${c === 'violet' ? 'teal' : c} card-lift`;
    card.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="text-2xl">${metric.icon || '📊'}</span>
      </div>
      <div>
        <p class="feat-value text-${c}-600">${pct}%</p>
        <p class="text-xs font-bold text-slate-500 mt-1">${metric.name || metric.label || 'Indicator'}</p>
      </div>
      <div class="feat-bar-track"><div class="feat-bar-fill" style="width:0%;background:var(--${c === 'violet' ? 'violet' : c})"></div></div>
    `;
    metricsGrid.appendChild(card);
    setTimeout(() => {
      const fill = card.querySelector('.feat-bar-fill');
      if (fill) fill.style.width = pct + '%';
    }, 100 + i * 80);
  });

  // ─ Metric summary banner ─
  const banner = document.getElementById('metrics-summary-banner');
  if (data.metrics_summary || data.summary) {
    banner.textContent = data.metrics_summary || data.summary;
    banner.style.display = 'block';
    banner.style.background  = isCataract ? 'rgba(225,29,72,.06)' : 'rgba(13,148,136,.06)';
    banner.style.borderColor = isCataract ? 'rgba(225,29,72,.2)'  : 'rgba(13,148,136,.2)';
    banner.style.color       = isCataract ? '#9f1239' : '#134e4a';
  } else {
    banner.style.display = 'none';
  }

  // ─ Explanations ─
  const simple   = data.explanation_simple    || data.simple_explanation    || '--';
  const technical= data.explanation_technical || data.technical_explanation || '--';
  const aiModel  = data.explanation_ai_model  || data.ai_model_explanation  || '--';
  document.getElementById('exp-simple').textContent    = simple;
  document.getElementById('exp-technical').textContent = technical;
  document.getElementById('exp-ai_model').textContent  = aiModel;

  // ─ AI Report ─
  const reportEl = document.getElementById('ai-report-content');
  const reportText = data.ai_report || data.report || null;
  if (reportText) {
    renderReport(reportEl, reportText);
  } else {
    // Generate locally via Groq
    reportEl.textContent = 'Generating report…';
    generateAIReport(data, groqKey).then(report => {
      renderReport(reportEl, report);
    }).catch(() => {
      reportEl.textContent = 'Report generation failed. Add your Groq key in Config.';
    });
  }

  // Show results
  resultsRoot.style.display = 'block';
  setTimeout(() => {
    resultsRoot.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  showToast('✓ Analysis complete', '#10b981');
}

function renderReport(el, text) {
  if (md) {
    el.classList.add('md-body');
    el.innerHTML = md.render(text);
  } else {
    el.textContent = text;
  }
}

async function generateAIReport(data, groqKey) {
  const verdict    = data.verdict || data.prediction || 'Unknown';
  const confidence = Math.round(parseFloat(data.confidence ?? data.avg_confidence ?? 0) * 100);
  const metrics    = (data.metrics || []).map(m => `${m.name}: ${Math.round((m.value || 0)*100)}%`).join(', ');
  const models     = (data.models  || []).map(m => `${m.name || 'Model'}: ${m.prediction || 'N/A'}`).join(', ');

  const prompt = `You are an expert ophthalmologist AI. Write a concise clinical report for a patient based on the following cataract detection results.

Verdict: ${verdict}
Confidence: ${confidence}%
Clinical Indicators: ${metrics || 'N/A'}
Model Votes: ${models || 'N/A'}

Write the report in professional medical language with sections: ## Summary, ## Clinical Findings, ## Recommendation. Use Markdown format.`;

  return await window.api.groq.summarize(prompt, groqKey);
}

// ── Loading overlay ────────────────────────────
function setLoading(show) {
  loadingOverlay.style.display = show ? 'flex' : 'none';
}

// ── Toast ──────────────────────────────────────
let toastTimer = null;
function showToast(message, color = '#0d9488') {
  toastText.textContent = message;
  const dot = resultToast.querySelector('.toast-dot');
  if (dot) dot.style.background = color;
  resultToast.style.display = 'flex';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { resultToast.style.display = 'none'; }, 3500);
}

// ── Subpage Drawer ─────────────────────────────
function openSubpage(tab) {
  subpageOverlay.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  switchDrawerTab(tab, null);
}

function closeSubpage() {
  subpageOverlay.style.display = 'none';
  document.body.style.overflow = '';
}

function switchDrawerTab(tab, clickedBtn) {
  // Update tab buttons
  document.querySelectorAll('.drawer-tab').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.trim().toLowerCase() === tab.toLowerCase());
  });
  if (clickedBtn) {
    document.querySelectorAll('.drawer-tab').forEach(b => b.classList.remove('active'));
    clickedBtn.classList.add('active');
  }

  // Update drawer pages
  document.querySelectorAll('.drawer-page').forEach(page => {
    page.classList.toggle('active', page.id === `dp-${tab}`);
  });
}

// Make globally accessible (called from inline onclick in HTML)
window.openSubpage     = openSubpage;
window.closeSubpage    = closeSubpage;
window.switchDrawerTab = switchDrawerTab;
window.saveSettings    = saveSettings;

// ── FAQ accordion ──────────────────────────────
function toggleFaq(headerEl) {
  const item = headerEl.closest('.faq-item');
  const isOpen = item.classList.contains('open');
  // Close all
  document.querySelectorAll('.faq-item.open').forEach(f => f.classList.remove('open'));
  // Open clicked if it wasn't open
  if (!isOpen) item.classList.add('open');
}
window.toggleFaq = toggleFaq;

// ── Explanation tabs ───────────────────────────
function showExpTab(tab, clickedBtn) {
  document.querySelectorAll('.exp-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.exp-panel').forEach(p => p.classList.remove('active'));
  clickedBtn.classList.add('active');
  const panel = document.getElementById(`exp-${tab}`);
  if (panel) panel.classList.add('active');
}
window.showExpTab = showExpTab;

// ── Chat FAB & Panel ───────────────────────────
function toggleChat() {
  chatPanel.classList.toggle('open');
  if (chatPanel.classList.contains('open') && chatMessages.children.length === 0) {
    addChatMessage('bot', "👁️ Hello! I'm your Medical AI. Ask me anything about cataracts or your results.");
  }
}
window.toggleChat = toggleChat;

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  const groqKey = getGroqKey();
  if (!groqKey) {
    addChatMessage('bot', '⚠️ Please save your Groq API key in Config first.');
    return;
  }

  addChatMessage('user', text);
  chatInput.value = '';

  const typingEl = addTypingIndicator();

  try {
    const lang = document.getElementById('chat-lang').value || 'English';

    // Build context-aware system prompt
    let systemPrompt = `You are a helpful Medical Assistant specialized in Cataract and Eye Health.
Respond in ${lang}.
Use simple, caring language.
If the user asks about surgery, mention that Ayushman Bharat offers free cataract treatment in India.
Always advise consulting a real ophthalmologist.`;

    if (lastDiagnostic) {
      const verdict = lastDiagnostic.verdict || lastDiagnostic.prediction || 'Unknown';
      const conf    = Math.round(parseFloat(lastDiagnostic.confidence ?? 0) * 100);
      systemPrompt += `\n\nPatient's last scan result: ${verdict} at ${conf}% confidence. Use this context when answering questions.`;
    }

    // Sliding window history
    chatHistory.push({ role: 'user', content: text });
    if (chatHistory.length > MAX_CHAT_HIST) chatHistory.shift();

    const fullMsg = chatHistory.map(m => `${m.role === 'user' ? 'Patient' : 'Doctor'}: ${m.content}`).join('\n');

    const reply = await window.api.groq.chat(fullMsg, lang, groqKey, systemPrompt);

    chatHistory.push({ role: 'assistant', content: reply });
    if (chatHistory.length > MAX_CHAT_HIST) chatHistory.shift();

    typingEl.remove();
    addChatMessage('bot', reply);
  } catch (err) {
    typingEl.remove();
    addChatMessage('bot', '⚠️ Failed to get a response. Check your connection or Groq API key.');
  }
});

function addChatMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg-anim flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

  const bubble = document.createElement('div');
  bubble.className = role === 'user'
    ? 'max-w-[80%] px-3.5 py-2.5 rounded-2xl rounded-tr-sm text-sm font-medium text-white shadow'
    : 'max-w-[85%] px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-sm text-slate-700 bg-white shadow border border-slate-100';
  if (role === 'user') bubble.style.background = 'linear-gradient(135deg,#0d9488,#2563eb)';

  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

function addTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-anim flex justify-start';
  wrap.innerHTML = `<div class="px-4 py-2.5 rounded-2xl rounded-tl-sm bg-white shadow border border-slate-100 flex gap-1 items-center">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>`;
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

// ── Init ───────────────────────────────────────
loadSettings();
console.log('Cataract Hub: renderer.js loaded ✓');

const { contextBridge, ipcRenderer } = require('electron');

try {
  contextBridge.exposeInMainWorld('api', {
    huggingface: {
      /**
       * Call a Gradio Space via the Main Process (standard IPC approach)
       * @param {string} spaceId - The HF Space ID
       * @param {string} apiName - The named endpoint (e.g., '/predict_ensemble')
       * @param {object} payload - Object containing input data (e.g., { image, groq_api_key })
       */
      call: async (spaceId, apiName, payload) => {
        console.log(`IPC Gradio Call: ${spaceId} [Endpoint: ${apiName}]`);
        return await ipcRenderer.invoke('gradio-call', { spaceId, apiName, payload });
      }
    },
    groq: {
      summarize: async (findings, apiKey) => {
        const url = 'https://api.groq.com/openai/v1/chat/completions';
        try {
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${apiKey}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model: "llama-3.3-70b-versatile",
              messages: [
                {
                  role: "system",
                  content: "You are a friendly AI Eye Assistant. Analyze the provided cataract screening findings and provide a clinical report."
                },
                {
                  role: "user",
                  content: findings
                }
              ],
              temperature: 0.0,
              max_tokens: 1500,
            })
          });
          
          if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`Groq HTTP ${response.status}: ${errBody}`);
          }
          
          const result = await response.json();
          return result.choices[0].message.content;
        } catch (error) {
            console.error('Groq API Error:', error.message);
            throw error;
        }
      }
    }
  });
  console.log("Preload script: @gradio/client bridge ready.");
} catch (err) {
  console.error("Preload script: Fatal initialization error:", err);
}

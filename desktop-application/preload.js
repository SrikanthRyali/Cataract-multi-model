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
      /**
       * Exact Prompt Parity Summarizer (as used in app.py logic)
       */
      summarize: async (prompt, apiKey) => {
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
                  role: "user",
                  content: prompt
                }
              ],
              temperature: 0.0,
              max_tokens: 2000,
            })
          });
          
          if (!response.ok) throw new Error(`Groq API Error: ${response.status}`);
          const result = await response.json();
          return result.choices[0].message.content;
        } catch (error) {
            console.error('Groq Summarize Error:', error.message);
            throw error;
        }
      },
      /**
       * Medical AI Chat Assistant (Multilingual)
       */
      chat: async (userText, language, apiKey) => {
        const url = 'https://api.groq.com/openai/v1/chat/completions';
        const systemPrompt = `You are a helpful Medical Assistant specialized in Cataract. 
Respond in ${language}. 
Use simple, caring language. 
If the user asks about surgery, mention that Ayushman Bharat offers free treatment in India. 
Always advise consulting a real ophthalmologist.`;

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
                { role: "system", content: systemPrompt },
                { role: "user", content: userText }
              ],
              temperature: 0.7,
              max_tokens: 1000,
            })
          });
          
          if (!response.ok) throw new Error(`Groq API Error: ${response.status}`);
          const result = await response.json();
          return result.choices[0].message.content;
        } catch (error) {
            console.error('Groq Chat Error:', error.message);
            throw error;
        }
      }
    }
  });
  console.log("Cataract Hub: Preload Bridge Synchronized.");
} catch (err) {
  console.error("Preload script Error:", err);
}

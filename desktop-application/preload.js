const { contextBridge } = require('electron');

// We use dynamic import for @gradio/client because it is an ESM-only package
let gradioClient = null;

try {
  contextBridge.exposeInMainWorld('api', {
    huggingface: {
      /**
       * Call a Gradio Space using the official @gradio/client
       * @param {string} spaceId - The HF Space ID
       * @param {string} apiName - The named endpoint (e.g., '/predict_ensemble')
       * @param {object} payload - Object containing input data (e.g., { image, groq_api_key })
       */
      call: async (spaceId, apiName, payload) => {
        console.log(`Gradio Client Call: ${spaceId} [Endpoint: ${apiName}]`);
        
        try {
          if (!gradioClient) {
            const { Client } = await import('@gradio/client');
            gradioClient = await Client.connect(spaceId);
          }
          
          // Official client handles polling, SSE, and versioning automatically
          const result = await gradioClient.predict(apiName, payload);
          console.log("Gradio Result Object:", result);
          return result.data;
        } catch (error) {
          console.error(`Gradio Client Error:`, error.message);
          throw error;
        }
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

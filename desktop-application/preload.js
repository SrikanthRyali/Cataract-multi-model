const { contextBridge, ipcRenderer } = require('electron');
const axios = require('axios');

contextBridge.exposeInMainWorld('api', {
  huggingface: {
    inference: async (repoId, imageData, token) => {
      const url = `https://api-inference.huggingface.co/models/${repoId}`;
      try {
        const response = await axios.post(url, imageData, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/octet-stream'
          },
          responseType: 'json'
        });
        return response.data;
      } catch (error) {
        console.error(`HF Inference Error (${repoId}):`, error.response?.data || error.message);
        throw error;
      }
    }
  },
  groq: {
    chat: async (messages, apiKey) => {
      const url = 'https://api.groq.com/openai/v1/chat/completions';
      try {
        const response = await axios.post(url, {
          model: "llama-3.3-70b-versatile",
          messages: messages,
          temperature: 0.0,
          max_tokens: 2000,
        }, {
            headers: {
              'Authorization': `Bearer ${apiKey}`,
              'Content-Type': 'application/json'
            }
        });
        return response.data.choices[0].message.content;
      } catch (error) {
          console.error('Groq API Error:', error.response?.data || error.message);
          throw error;
      }
    }
  }
});

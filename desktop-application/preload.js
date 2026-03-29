const { contextBridge } = require('electron');
const axios = require('axios');

contextBridge.exposeInMainWorld('api', {
  huggingface: {
    /**
     * Call the Gradio Space API for the ensemble prediction.
     * @param {string} spaceId - The HF Space ID (e.g., 'Srikanth22MH1A42C6/models-api')
     * @param {string} imageBase64 - The base64 string of the eye image
     * @param {string} groqKey - Optional Groq key to pass to the space
     */
    predict: async (spaceId, imageBase64, groqKey) => {
      // Slugify space ID to get the subdomain: user/space -> user-space
      const slug = spaceId.replace('/', '-').toLowerCase();
      const url = `https://${slug}.hf.space/run/predict`;
      
      try {
        const response = await axios.post(url, {
          data: [
            imageBase64, // Input image (data:image/...)
            groqKey || "" // Optional Groq key
          ]
        }, {
          headers: { 'Content-Type': 'application/json' },
          timeout: 45000 // ML models can take time to wake up/run
        });
        
        // Gradio returns array in response.data.data
        return response.data.data;
      } catch (error) {
        console.error('Space API Error:', error.response?.data || error.message);
        throw error;
      }
    }
  },
  groq: {
    summarize: async (findings, apiKey) => {
      const url = 'https://api.groq.com/openai/v1/chat/completions';
      try {
        const response = await axios.post(url, {
          model: "llama-3.3-70b-versatile",
          messages: [
            {
              role: "system",
              content: "You are a friendly AI Eye Assistant. Analyze the provided cataract screening findings and provide a brief clinical report."
            },
            {
              role: "user",
              content: findings
            }
          ],
          temperature: 0.0,
          max_tokens: 1500,
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

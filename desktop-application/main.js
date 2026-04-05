const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    title: "Cataract Hub — Clinical Eye Analysis",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.setMenu(null);
  win.loadFile('index.html');
  // win.webContents.openDevTools();
}

let gradioClient = null;

app.whenReady().then(() => {
  createWindow();

  // IPC handler for Gradio calls
  ipcMain.handle('gradio-call', async (event, { spaceId, apiName, payload }) => {
    try {
      if (!gradioClient) {
        const { Client } = await import('@gradio/client');
        console.log(`Connecting to Space: ${spaceId}...`);
        try {
          // Use hf_token if provided for authentication
          const clientOptions = payload && payload.hfToken ? { hf_token: payload.hfToken } : {};
          gradioClient = await Client.connect(spaceId, clientOptions);
        } catch (connErr) {
          const directUrl = `https://${spaceId.replace('/', '-').toLowerCase()}.hf.space`;
          const clientOptions = payload && payload.hfToken ? { hf_token: payload.hfToken } : {};
          gradioClient = await Client.connect(directUrl, clientOptions);
        }
      }
      
      // Node.js Gradio client expects Buffers for images
      if (payload.image && typeof payload.image === 'string' && payload.image.startsWith('data:')) {
        const base64Data = payload.image.split(',')[1];
        payload.image = Buffer.from(base64Data, 'base64');
      }

      // Remove hfToken from payload before prediction (it's for connection only)
      delete payload.hfToken;

      const result = await gradioClient.predict(apiName, payload);
      return result.data;
    } catch (error) {
      console.error('Main Process Gradio Error:', error.message);
      throw error;
    }
  });

  // IPC handler for Medical AI Chat
  ipcMain.handle('chat-call', async (event, { message, language, groqApiKey }) => {
    try {
      if (!gradioClient) {
        const { Client } = await import('@gradio/client');
        gradioClient = await Client.connect("Srikanth22MH1A42C6/model-api-2");
      }
      const result = await gradioClient.predict('/chat', {
        message: message,
        language: language,
        groq_api_key: groqApiKey || ""
      });
      return { reply: result.data[0] };
    } catch (error) {
      console.error('Main Process Chat Error:', error.message);
      throw error;
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

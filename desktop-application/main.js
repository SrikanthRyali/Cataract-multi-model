const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    title: "Cataract Hub — Clinical Eye Analysis",
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

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
        // Dynamic import because @gradio/client is ESM
        const { Client } = await import('@gradio/client');
        
        // Detailed logging to help debug "Space metadata could not be loaded"
        console.log(`Connecting to Space: ${spaceId}...`);
        
        try {
          // Attempt connecting via the short spaceId
          gradioClient = await Client.connect(spaceId);
        } catch (connErr) {
          console.warn(`Connection to Space ID '${spaceId}' failed:`, connErr.message);
          
          // Secondary Attempt: Try direct URL if ID discovery fails
          const directUrl = `https://${spaceId.replace('/', '-').toLowerCase()}.hf.space`;
          console.log(`Falling back to direct URL: ${directUrl}...`);
          gradioClient = await Client.connect(directUrl);
        }
      }
      
      // Node.js Gradio client expects Buffers for images when passed as bit-data
      if (payload.image && typeof payload.image === 'string' && payload.image.startsWith('data:')) {
        const base64Data = payload.image.split(',')[1];
        payload.image = Buffer.from(base64Data, 'base64');
      }

      // Ensure optional parameters are provided at least as empty strings
      if (payload.groq_api_key === undefined || payload.groq_api_key === null) {
          payload.groq_api_key = "";
      }

      const result = await gradioClient.predict(apiName, payload);
      return result.data;
    } catch (error) {
      console.error('Main Process Gradio Error:', error.message);
      // Pass the full error details back to the renderer for debugging
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

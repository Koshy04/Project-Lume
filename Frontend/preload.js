const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    runPython: (mode, audioMode, ttsEngine, llmEngine, visionEnabled) => ipcRenderer.send('run-python', mode, audioMode, ttsEngine, llmEngine, visionEnabled),
    stopPython: () => ipcRenderer.send('stop-python'),
    setScreenshotRegion: (region) => ipcRenderer.send('set-screenshot-region', region),
    toggleVision: (keepModelInMemory) => ipcRenderer.send('toggle-vision', keepModelInMemory),
    onPythonOutput: (callback) => ipcRenderer.on('python-output', (_event, data) => callback(data)),
    onTtsEnginesUpdate: (callback) => ipcRenderer.on('update-tts-engines', (_event, engines) => callback(engines)),
    onLlmEnginesUpdate: (callback) => ipcRenderer.on('update-llm-engines', (_event, engines) => callback(engines)),
    onVisionFeedUpdate: (callback) => ipcRenderer.on('vision-feed-update', (_event, data) => callback(data)),
    onPythonExit: (callback) => ipcRenderer.on('python-exit', () => callback()),
    onScreenDimensions: (callback) => ipcRenderer.on('screen-dimensions', (_event, dimensions) => callback(dimensions)),
});
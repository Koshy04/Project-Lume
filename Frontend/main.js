const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function getPythonExecutable() {
    const projectRoot = path.join(__dirname, '..');
    const pythonFile = process.platform === 'win32' ? 'python.exe' : 'python';
    return path.join(projectRoot, '.venv', 'Scripts', pythonFile);
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 700,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    mainWindow.loadFile('index.html');

    // Send screen dimensions to the renderer process once it's loaded
    mainWindow.webContents.on('did-finish-load', () => {
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.size;
        mainWindow.webContents.send('screen-dimensions', { width, height });
    });
}

app.whenReady().then(() => {
    createWindow();
    const pythonExecutable = getPythonExecutable();
    const backendPath = path.join(__dirname, '..', 'Backend');
    const discoverScript = path.join(backendPath, 'bridge', 'discover_engines.py');

    console.log(`--- Running Engine Discovery ---`);
    const discoveryProcess = spawn('cmd.exe', ['/c', pythonExecutable, discoverScript], { cwd: backendPath });
    let stdout = '';
    discoveryProcess.stdout.on('data', (data) => { stdout += data.toString(); });
    discoveryProcess.stderr.on('data', (data) => {
        console.error(`Discovery STDERR: ${data.toString()}`);
        mainWindow.webContents.send('python-output', `Discovery Error: ${data.toString()}`);
    });
    discoveryProcess.on('close', (code) => {
        console.log(`Discovery process exited with code ${code}`);
        if (code === 0) {
            const lines = stdout.split('\n');
            const ttsLine = lines.find(line => line.startsWith('TTS_ENGINES:'));
            const ttsEngines = ttsLine ? ttsLine.replace('TTS_ENGINES:', '').trim().split(',').filter(e => e) : [];
            mainWindow.webContents.send('update-tts-engines', ttsEngines);
            
            const llmLine = lines.find(line => line.startsWith('LLM_ENGINES:'));
            const llmEngines = llmLine ? llmLine.replace('LLM_ENGINES:', '').trim().split(',').filter(e => e) : [];
            mainWindow.webContents.send('update-llm-engines', llmEngines);
        } else {
            mainWindow.webContents.send('update-tts-engines', []);
            mainWindow.webContents.send('update-llm-engines', []);
        }
    });
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

ipcMain.on('run-python', (event, mode, audioMode, ttsEngine, llmEngine, visionEnabled) => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }

    const pythonExecutable = getPythonExecutable();
    const backendPath = path.join(__dirname, '..', 'Backend');
    const mainScript = path.join(backendPath, 'main_app.py');

    const scriptArgs = [
        '-u', 
        mainScript,
        mode,
        audioMode,
        ttsEngine,
        llmEngine,
        visionEnabled ? 'true' : 'false'
    ];

    console.log(`--- Starting Main App (Unbuffered): ${pythonExecutable} ${scriptArgs.join(' ')} ---`);
    pythonProcess = spawn(pythonExecutable, scriptArgs, { cwd: backendPath, stdio: ['pipe', 'pipe', 'pipe'] });

    pythonProcess.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        lines.forEach(line => {
            if (line.startsWith('VISION_FEED:')) {
                const base64Data = line.substring('VISION_FEED:'.length);
                if (mainWindow) {
                    mainWindow.webContents.send('vision-feed-update', base64Data);
                }
            } else if (line.trim()) {
                if (mainWindow) {
                    mainWindow.webContents.send('python-output', line);
                }
            }
        });
    });

    pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        console.error(`Python stderr: ${message}`);
        mainWindow.webContents.send('python-output', `STDERR: ${message}`);
    });
    pythonProcess.on('close', (code, signal) => {
        console.log(`Python script finished with code ${code} and signal ${signal}`);
        mainWindow.webContents.send('python-exit');
        if (code !== 0 && signal !== 'SIGTERM') {
            mainWindow.webContents.send('python-output', 'Process exited unexpectedly.');
        }
        pythonProcess = null;
    });
});

ipcMain.on('toggle-vision', (event, keepModelInMemory) => {
    if (pythonProcess && pythonProcess.stdin) {
        const command = `TOGGLE_VISION:${String(keepModelInMemory)}\n`;
        console.log(`Sending command to Python stdin: "${command.trim()}"`);
        pythonProcess.stdin.write(command);
    } else {
        console.warn("Received 'toggle-vision' command but no Python process is running or stdin is not available.");
    }
});

ipcMain.on('set-screenshot-region', (event, region) => {
    if (pythonProcess && pythonProcess.stdin) {
        // Convert null to an empty string or a specific command
        const regionString = region ? JSON.stringify(region) : 'null';
        const command = `SET_REGION:${regionString}\n`;
        console.log(`Sending command to Python stdin: "${command.trim()}"`);
        pythonProcess.stdin.write(command);
    }
});

app.on('window-all-closed', () => {
    if (pythonProcess) pythonProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    if (pythonProcess) pythonProcess.kill();
});

ipcMain.on('stop-python', () => {
    if (pythonProcess) {
        console.log('Attempting graceful shutdown of Python process...');
        mainWindow.webContents.send('python-output', 'Attempting graceful shutdown...');
        
        try {
            pythonProcess.stdin.write("SHUTDOWN\n");

            const shutdownTimeout = setTimeout(() => {
                if (pythonProcess) {
                    console.warn('Graceful shutdown timed out. Forcing termination.');
                    mainWindow.webContents.send('python-output', 'Graceful shutdown timed out. Forcing process kill.');
                    pythonProcess.kill('SIGKILL');
                    pythonProcess = null;
                }
            }, 5000); // 5 seconds

            pythonProcess.on('exit', () => {
                clearTimeout(shutdownTimeout); // Cancel the failsafe kill if it exits cleanly
                console.log('Python process exited gracefully.');
                mainWindow.webContents.send('python-output', 'Process stopped successfully.');
                pythonProcess = null;
            });

        } catch (error) {
            console.error('Failed to send shutdown command. Forcing kill.', error);
            pythonProcess.kill('SIGKILL'); // Force kill if stdin is not writable
            pythonProcess = null;
        }

    } else {
        mainWindow.webContents.send('python-output', 'No process is currently running.');
    }
});
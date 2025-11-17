document.addEventListener('DOMContentLoaded', () => {
    // --- Global State ---
    let visionEnabled = false;
    let keepModelInMemory = true;
    let isProcessRunning = false;
    let screenDimensions = { width: 1920, height: 1080 }; // Default
    let screenshotRegion = null;

    // --- Element References ---
    const localPttBtn = document.getElementById('local-ptt-btn');
    const localVadBtn = document.getElementById('local-vad-btn');
    const discordBtn = document.getElementById('discord-btn');
    const stopBtn = document.getElementById('stop-btn');
    const outputDiv = document.getElementById('output');
    const ttsEngineSelect = document.getElementById('tts-engine-select');
    const llmEngineSelect = document.getElementById('llm-engine-select');

    // Sidebar and page elements
    const toggleBtn = document.querySelector('.toggle-btn');
    const mainContent = document.getElementById('main');
    const sidebar = document.getElementById('sidebar');
    const homeLink = document.getElementById('home-link');
    const visionLink = document.getElementById('vision-link');
    const homePage = document.getElementById('home-page');
    const visionPage = document.getElementById('vision-page');
    const visionToggle = document.getElementById('vision-toggle');
    const visionMemoryToggle = document.getElementById('vision-memory-toggle');
    const visionFeedImg = document.getElementById('vision-feed-img');
    const visionFeedContainer = document.getElementById('vision-feed-container');

    // Region selection elements
    const setRegionBtn = document.getElementById('set-region-btn');
    const clearRegionBtn = document.getElementById('clear-region-btn');
    const regionOverlay = document.getElementById('region-overlay');
    const selectionBox = document.getElementById('selection-box');
    const regionStatus = document.getElementById('region-status');
    function ansiToHtml(text) {
        const ansiColorMap = {
            '0': 'inherit',
            '30': '#000000', // Black
            '31': '#FF0000', // Red
            '32': '#00FF00', // Green
            '33': '#FFFF00', // Yellow
            '34': '#0000FF', // Blue
            '35': '#FF00FF', // Magenta
            '36': '#00FFFF', // Cyan
            '37': '#FFFFFF', // White
            '90': '#808080', // Bright Black (Gray)
            '91': '#FF5555', // Bright Red
            '92': '#55FF55', // Bright Green
            '93': '#FFFF55', // Bright Yellow
            '94': '#5555FF', // Bright Blue
            '95': '#FF55FF', // Bright Magenta
            '96': '#55FFFF', // Bright Cyan
            '97': '#FFFFFF'  // Bright White
        };

        const regex = /\u001b\[(\d+)(;(\d+))*m/g;
        let openSpan = false;

        return text.replace(regex, (match, code) => {
            let color = ansiColorMap[code];
            if (color) {
                let style = `color:${color}`;
                let result = '';
                if (openSpan) {
                    result += '</span>';
                }
                result += `<span style="${style}">`;
                openSpan = true;
                return result;
            } else if (code === '0') {
                let result = '';
                if (openSpan) {
                    result = '</span>';
                    openSpan = false;
                }
                return result;
            }
            return match;
        }) + (openSpan ? '</span>' : ''); 
    }

    function toggleSidebar() {
        mainContent.classList.toggle('sidebar-open');
        sidebar.style.left = mainContent.classList.contains('sidebar-open') ? '0' : '-250px';
    }

    /** Handles switching between pages. */
    function switchPage(activeLink, activePage) {
        document.querySelectorAll('.page-content').forEach(page => page.style.display = 'none');
        document.querySelectorAll('#sidebar a').forEach(link => link.classList.remove('active'));
        activePage.style.display = 'block';
        activeLink.classList.add('active');
    }
    
    // --- Event Listeners ---
    toggleBtn.addEventListener('click', toggleSidebar);
    homeLink.addEventListener('click', () => switchPage(homeLink, homePage));
    visionLink.addEventListener('click', () => switchPage(visionLink, visionPage));
    
    visionToggle.addEventListener('change', (event) => {
        visionEnabled = event.target.checked;
        console.log(`Vision state set to: ${visionEnabled}`);
        if (isProcessRunning) {
            console.log(`Sending toggle command. Keep model in memory: ${keepModelInMemory}`);
            window.electronAPI.toggleVision(keepModelInMemory);
        }
    });

    visionMemoryToggle.addEventListener('change', (event) => {
        keepModelInMemory = event.target.checked;
        console.log(`Keep vision model in memory set to: ${keepModelInMemory}`);
    });

    // --- Region Selection Logic ---
    let isDrawing = false;
    let startX, startY;

    function updateRegionStatus() {
        if (screenshotRegion) {
            regionStatus.textContent = `Active Region: ${screenshotRegion.width}x${screenshotRegion.height} at (${screenshotRegion.left}, ${screenshotRegion.top})`;
            clearRegionBtn.disabled = false;
        } else {
            regionStatus.textContent = 'Active Region: Full Screen';
            clearRegionBtn.disabled = true;
        }
    }

    setRegionBtn.addEventListener('click', () => {
        regionOverlay.style.display = 'block';
        visionFeedContainer.style.cursor = 'crosshair';
    });

    clearRegionBtn.addEventListener('click', () => {
        screenshotRegion = null;
        selectionBox.style.display = 'none';
        window.electronAPI.setScreenshotRegion(null);
        updateRegionStatus();
    });

    regionOverlay.addEventListener('mousedown', (e) => {
        isDrawing = true;
        startX = e.offsetX;
        startY = e.offsetY;
        selectionBox.style.left = `${startX}px`;
        selectionBox.style.top = `${startY}px`;
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        selectionBox.style.display = 'block';
    });

    regionOverlay.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const currentX = e.offsetX;
        const currentY = e.offsetY;

        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);
        const left = Math.min(currentX, startX);
        const top = Math.min(currentY, startY);

        selectionBox.style.width = `${width}px`;
        selectionBox.style.height = `${height}px`;
        selectionBox.style.left = `${left}px`;
        selectionBox.style.top = `${top}px`;
    });

    regionOverlay.addEventListener('mouseup', (e) => {
        if (!isDrawing) return;
        isDrawing = false;
        regionOverlay.style.display = 'none';
        visionFeedContainer.style.cursor = 'default';

        const imgNaturalWidth = visionFeedImg.naturalWidth;
        const imgNaturalHeight = visionFeedImg.naturalHeight;

        if (imgNaturalWidth === 0 || imgNaturalHeight === 0) return;

        const scaleX = screenDimensions.width / regionOverlay.clientWidth;
        const scaleY = screenDimensions.height / regionOverlay.clientHeight;
        
        screenshotRegion = {
            left: Math.round(parseInt(selectionBox.style.left) * scaleX),
            top: Math.round(parseInt(selectionBox.style.top) * scaleY),
            width: Math.round(parseInt(selectionBox.style.width) * scaleX),
            height: Math.round(parseInt(selectionBox.style.height) * scaleY)
        };
        window.electronAPI.setScreenshotRegion(screenshotRegion);
        updateRegionStatus();
    });
    /** Populates a dropdown with a list of engines. */
    function populateDropdown(selectElement, engines) {
        selectElement.innerHTML = '';
        const hasEngines = engines && engines.length > 0;

        if (hasEngines) {
            engines.forEach(engine => {
                const option = document.createElement('option');
                option.value = engine;
                option.textContent = engine.charAt(0).toUpperCase() + engine.slice(1);
                selectElement.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.textContent = "No engines found";
            option.disabled = true;
            selectElement.appendChild(option);
        }
        localPttBtn.disabled = !hasEngines;
        localVadBtn.disabled = !hasEngines;
        discordBtn.disabled = !hasEngines;
    }

    // --- IPC Listeners from Main Process ---
    window.electronAPI.onScreenDimensions((dimensions) => {
        console.log('Received screen dimensions:', dimensions);
        screenDimensions = dimensions;
    });

    window.electronAPI.onTtsEnginesUpdate((engines) => populateDropdown(ttsEngineSelect, engines));
    window.electronAPI.onLlmEnginesUpdate((engines) => populateDropdown(llmEngineSelect, engines));
    window.electronAPI.onPythonExit(() => {
        console.log("Received python-exit event.");
        isProcessRunning = false;
    });

    window.electronAPI.onPythonOutput((data) => {
        const p = document.createElement('p');
        p.innerHTML = ansiToHtml(data.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
        outputDiv.appendChild(p);
        outputDiv.scrollTop = outputDiv.scrollHeight;
    });

    window.electronAPI.onVisionFeedUpdate((base64Data) => {
        if (visionFeedImg) {
            visionFeedImg.src = `data:image/jpeg;base64,${base64Data}`;
        }
    });

    function startPythonProcess(mode, audioMode) {
        isProcessRunning = true;
        outputDiv.textContent = `Starting ${audioMode} Mode...\n`;
        const selectedTtsEngine = ttsEngineSelect.value;
        const selectedLlmEngine = llmEngineSelect.value;

        console.log(`[UI] Sending run-python command. Vision on startup: ${visionEnabled}`);
        window.electronAPI.runPython(mode, audioMode, selectedTtsEngine, selectedLlmEngine, visionEnabled);
    }

    localPttBtn.addEventListener('click', () => startPythonProcess('1', 'ptt'));
    localVadBtn.addEventListener('click', () => startPythonProcess('1', 'vad'));
    discordBtn.addEventListener('click', () => startPythonProcess('2', 'discord'));
    stopBtn.addEventListener('click', () => {
        isProcessRunning = false;
        window.electronAPI.stopPython();
    });

    // Initial UI state
    updateRegionStatus();
});
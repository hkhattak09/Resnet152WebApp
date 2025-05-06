const COMPRESSED_MODEL_URL = './ag_modelQ.onnx.gz';
const MODEL_KEY = 'ag_modelQ';
const MODEL_VERSION = 1;
const DB_NAME = 'ModelCacheDB';
const DB_VERSION = 1;
const STORE_NAME = 'models';

const IMAGE_SIZE = 224;
const NORM_MEAN = [0.485, 0.456, 0.406];
const NORM_STD_DEV = [0.229, 0.224, 0.225];


const webcamFeed = document.getElementById('webcamFeed');
const predictButton = document.getElementById('predictButton');
const buttonText = document.getElementById('buttonText');
const statusDiv = document.getElementById('status');
const statusText = document.getElementById('statusText');
const preprocessedCanvas = document.getElementById('preprocessedPreviewCanvas');
const ageResultSpan = document.getElementById('ageResult');
const genderResultSpan = document.getElementById('genderResult');
const loadTimeValueSpan = document.getElementById('loadTimeValue');
const latencyValueSpan = document.getElementById('latencyValue');

const webcamOverlayCanvas = document.getElementById('webcamOverlayCanvas');
const overlayCtx = webcamOverlayCanvas.getContext('2d');

const ctx = preprocessedCanvas.getContext('2d'); 


let ortSession = null;
let isWebcamActive = false;
let db = null;


function setStatus(message, type = 'idle') {
    statusText.textContent = message;
    statusDiv.className = ''; 
    statusDiv.classList.add(`status-${type}`);
    console.log(`Status (${type}): ${message}`);
}


function updateButtonState() {
    if (ortSession && isWebcamActive) {
        predictButton.disabled = false;
        buttonText.textContent = 'Predict Age & Gender';
    } else {
        predictButton.disabled = true;
        if (!ortSession && !predictButton.textContent.includes("Failed")) {
             buttonText.textContent = statusText.textContent;
        } else if (!isWebcamActive && ortSession) {
            buttonText.textContent = 'Starting Webcam...';
        } else if (!isWebcamActive && !ortSession && !predictButton.textContent.includes("Failed")) {
             buttonText.textContent = 'Initializing...';
        }
    }
}


function openDB() {
    return new Promise((resolve, reject) => {
        if (db) { resolve(db); return; }
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = (event) => reject("IndexedDB error: " + event.target.error);
        request.onsuccess = (event) => { db = event.target.result; console.log("IndexedDB opened successfully."); resolve(db); };
        request.onupgradeneeded = (event) => {
            console.log("IndexedDB upgrade needed.");
            db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: 'key' });
                console.log(`Object store "${STORE_NAME}" created.`);
            }
        };
    });
}
function getItem(key) {
    return new Promise(async (resolve, reject) => {
        try {
            const dbInstance = await openDB();
            const transaction = dbInstance.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.get(key);
            request.onerror = (event) => reject("Error getting item from DB: " + event.target.error);
            request.onsuccess = (event) => resolve(event.target.result);
        } catch (error) { reject(error); }
    });
}
function setItem(key, value) {
    return new Promise(async (resolve, reject) => {
        try {
            const dbInstance = await openDB();
            const transaction = dbInstance.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const item = { key: key, data: value.data, version: value.version };
            const request = store.put(item);
            request.onerror = (event) => reject("Error setting item in DB: " + event.target.error);
            request.onsuccess = () => resolve();
        } catch (error) { reject(error); }
    });
}


function formatTime(milliseconds) {
    if (milliseconds < 0) return "N/A";
    if (milliseconds < 1000) {
        return `${milliseconds.toFixed(0)} ms`;
    } else {
        return `${(milliseconds / 1000).toFixed(2)} s`;
    }
}


async function loadModel() {
    const startTime = performance.now();
    setStatus('Initializing model...', 'loading');
    loadTimeValueSpan.textContent = 'Calculating...';

    try {
        if (typeof ort === 'undefined') throw new Error("ONNX Runtime Web library (ort) is not loaded.");
        if (typeof DecompressionStream === 'undefined') throw new Error("Browser does not support DecompressionStream API.");

        setStatus('Checking model cache...', 'loading');
        updateButtonState();
        let modelDataBuffer = null;
        let loadedFromCache = false;

        try {
            const cachedItem = await getItem(MODEL_KEY);
            if (cachedItem && cachedItem.version === MODEL_VERSION && cachedItem.data instanceof ArrayBuffer) {
                setStatus('Loading model from cache...', 'loading');
                console.log(`Found valid cached model version ${MODEL_VERSION}.`);
                modelDataBuffer = cachedItem.data;
                loadedFromCache = true;
            } else {
                if (cachedItem) console.log(`Cached model version mismatch or data invalid. Need version ${MODEL_VERSION}. Redownloading.`);
                else console.log("No cached model found. Downloading.");

                setStatus('Downloading model...', 'loading');
                updateButtonState();
                const downloadStartTime = performance.now();
                const response = await fetch(COMPRESSED_MODEL_URL);
                if (!response.ok) throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
                if (!response.body) throw new Error("Response body is missing.");
                console.log(`Download started... (${(performance.now() - downloadStartTime).toFixed(0)} ms)`);

                setStatus('Decompressing model...', 'loading');
                updateButtonState();
                const decompressStartTime = performance.now();
                const ds = new DecompressionStream('gzip');
                const decompressedStream = response.body.pipeThrough(ds);
                modelDataBuffer = await new Response(decompressedStream).arrayBuffer();
                console.log(`Model downloaded and decompressed (${(modelDataBuffer.byteLength / (1024 * 1024)).toFixed(2)} MB) in ${(performance.now() - downloadStartTime).toFixed(0)} ms (Decompression: ${(performance.now() - decompressStartTime).toFixed(0)} ms).`);

                setStatus('Saving model to cache...', 'loading');
                updateButtonState();
                const cacheStartTime = performance.now();
                await setItem(MODEL_KEY, { data: modelDataBuffer, version: MODEL_VERSION });
                console.log(`Model version ${MODEL_VERSION} saved to IndexedDB in ${(performance.now() - cacheStartTime).toFixed(0)} ms.`);
            }
        } catch (dbError) {
             console.error("IndexedDB access error during load/save:", dbError);
             setStatus(`Cache Error: ${dbError}. Refresh may help.`, 'error');
             throw new Error(`IndexedDB failed: ${dbError}`);
        }

        setStatus('Creating inference session...', 'loading');
        updateButtonState();
        if (!modelDataBuffer) throw new Error("Model data buffer is not available.");

        const sessionCreateStart = performance.now();
        ortSession = await ort.InferenceSession.create(modelDataBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log(`ONNX session created successfully from ${loadedFromCache ? 'cache' : 'download'} in ${(performance.now() - sessionCreateStart).toFixed(0)} ms.`);

        const endTime = performance.now();
        const totalLoadTime = endTime - startTime;
        loadTimeValueSpan.textContent = formatTime(totalLoadTime);
        console.log(`Total model load time: ${formatTime(totalLoadTime)}`);

        updateButtonState();
        return true;

    } catch (error) {
        console.error("Error during model initialization:", error);
        setStatus(`Model Init Error: ${error.message}. Refresh may help.`, 'error');
        ortSession = null;
        predictButton.disabled = true;
        buttonText.textContent = 'Model Init Failed';
        loadTimeValueSpan.textContent = 'Error';
        updateButtonState();
        return false;
    }
}

function drawWebcamOverlay() {
    if (!overlayCtx || !webcamOverlayCanvas || !webcamFeed.videoWidth || webcamFeed.videoWidth === 0) {
        if (webcamFeed.readyState >= 2) { 
             requestAnimationFrame(drawWebcamOverlay);
        }
        return;
    }
    webcamOverlayCanvas.width = webcamFeed.videoWidth;
    webcamOverlayCanvas.height = webcamFeed.videoHeight;
    overlayCtx.clearRect(0, 0, webcamOverlayCanvas.width, webcamOverlayCanvas.height);

    const boxWidthPercent = 0.60 / 1.8;
    const boxHeightPercent = 0.75 / 1.5;
    
    const boxWidth = webcamOverlayCanvas.width * boxWidthPercent;
    const boxHeight = webcamOverlayCanvas.height * boxHeightPercent;
    
    const boxX = (webcamOverlayCanvas.width - boxWidth) / 2;
    const boxY = (webcamOverlayCanvas.height - boxHeight) / 2 * 0.9; 

    overlayCtx.strokeStyle = 'lime'; 
    overlayCtx.lineWidth = 3;        
    overlayCtx.strokeRect(boxX, boxY, boxWidth, boxHeight);
}


async function startWebcam() {
    setStatus('Requesting camera access...', 'loading');
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Webcam access (getUserMedia) is not supported.');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
        });
        webcamFeed.srcObject = stream;
        await new Promise((resolve, reject) => {
            webcamFeed.onloadedmetadata = () => {
                webcamFeed.play().then(() => {
                    drawWebcamOverlay(); 
                    window.addEventListener('resize', drawWebcamOverlay);
                    resolve();
                }).catch(reject);
            };
            webcamFeed.onerror = (e) => reject(new Error("Webcam metadata error or play error."));
        });
        isWebcamActive = true;
        console.log('Webcam stream active.');
        updateButtonState();
        return true;
    } catch (error) {
        console.error("Error accessing webcam:", error);
        setStatus(`Webcam Error: ${error.message}. Check permissions.`, 'error');
        isWebcamActive = false;
        predictButton.disabled = true;
        buttonText.textContent = 'Webcam Failed';
        updateButtonState();
        return false;
    }
}


function preprocessVideoFrame(videoElement) {
    ctx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;
    const aspectRatio = videoWidth / videoHeight;
    let sourceX = 0, sourceY = 0, sourceWidth = videoWidth, sourceHeight = videoHeight;
    if (aspectRatio > 1) {
        sourceWidth = videoHeight; 
        sourceX = (videoWidth - sourceWidth) / 2;
    } else { 
        sourceHeight = videoWidth; 
        sourceY = (videoHeight - sourceHeight) / 2;
    }
    ctx.save();
    ctx.translate(IMAGE_SIZE, 0); 
    ctx.scale(-1, 1);             
    ctx.drawImage(videoElement,
                  sourceX, sourceY, sourceWidth, sourceHeight,
                  0, 0, IMAGE_SIZE, IMAGE_SIZE);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = imageData.data;
    const tensorData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);
    let tensorIndex = 0;
    for (let c = 0; c < 3; ++c) { 
        const mean = NORM_MEAN[c];
        const stdDev = NORM_STD_DEV[c];
        for (let i = c; i < data.length; i += 4) { 
            tensorData[tensorIndex++] = (data[i] / 255.0 - mean) / stdDev;
        }
    }
    return tensorData;
}


async function runInference(preprocessedData) {
    if (!ortSession) throw new Error("ONNX session is not loaded.");
    const inputName = ortSession.inputNames[0];
    const inputTensor = new ort.Tensor('float32', preprocessedData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
    const feeds = { [inputName]: inputTensor };

    const startTime = performance.now();
    const results = await ortSession.run(feeds);
    const endTime = performance.now();
    const latency = endTime - startTime;

    const ageOutputName = ortSession.outputNames.find(name => name.toLowerCase().includes('age')) || ortSession.outputNames[0];
    const genderOutputName = ortSession.outputNames.find(name => name.toLowerCase().includes('gender')) || ortSession.outputNames[1];

    if (!results[ageOutputName]) throw new Error(`Output tensor '${ageOutputName}' not found in results.`);
    if (!results[genderOutputName]) throw new Error(`Output tensor '${genderOutputName}' not found in results.`);

    return {
        ageTensor: results[ageOutputName],
        genderTensor: results[genderOutputName],
        inferenceLatency: latency
    };
}

function postprocessOutput(results) {
    const { ageTensor, genderTensor } = results;
    if (!ageTensor || !genderTensor) throw new Error("Missing output tensor(s) for postprocessing.");
    const ageValue = ageTensor.data[0];
    const genderProb = genderTensor.data[0];
    const predictedGender = genderProb > 0.5 ? 'Female' : 'Male';
    const predictedAge = Math.max(0, Math.round(ageValue));
    return { age: predictedAge, gender: predictedGender };
}

predictButton.addEventListener('click', async () => {
    if (!ortSession || !isWebcamActive) {
        setStatus("System not ready. Wait for initialization.", 'error');
        return;
    }
    predictButton.disabled = true;
    buttonText.textContent = 'Processing...';
    resetPredictionUI();
    setStatus('Capturing & Preprocessing...', 'processing');
    latencyValueSpan.textContent = 'Running...';

    try {
         if (!webcamFeed.videoWidth || !webcamFeed.videoHeight) {
             throw new Error("Webcam feed not ready or dimensions unavailable.");
         }

        const tensor = preprocessVideoFrame(webcamFeed);

        setStatus('Running inference...', 'processing');
        const { ageTensor, genderTensor, inferenceLatency } = await runInference(tensor);
        latencyValueSpan.textContent = formatTime(inferenceLatency);
        console.log(`Inference Latency: ${formatTime(inferenceLatency)}`);


        setStatus('Postprocessing...', 'processing');
        const { age, gender } = postprocessOutput({ ageTensor, genderTensor });

        ageResultSpan.textContent = age;
        genderResultSpan.textContent = gender;
        setStatus('Prediction complete! Ready for next.', 'success');
        console.log("Prediction finished successfully.");

    } catch (error) {
        console.error("Prediction failed:", error);
        setStatus(`Prediction Error: ${error.message}`, 'error');
        latencyValueSpan.textContent = 'Error';
    } finally {
        if (ortSession && isWebcamActive) {
             predictButton.disabled = false;
             buttonText.textContent = 'Predict Age & Gender';
        } else {
             updateButtonState();
        }
    }
});


function resetPredictionUI() {
    ageResultSpan.textContent = '-';
    genderResultSpan.textContent = '-';
    latencyValueSpan.textContent = '-';
}

async function initializeApp() {
    console.log("Initializing application...");
    setStatus('Initializing...', 'idle');
    loadTimeValueSpan.textContent = '-';
    latencyValueSpan.textContent = '-';
    updateButtonState();

    const modelLoaded = await loadModel();
    let webcamStarted = false;

    if (modelLoaded) {
        webcamStarted = await startWebcam();
    }

    if (modelLoaded && webcamStarted) {
        setStatus('Ready. Press Predict.', 'active');
    } else if (modelLoaded && !webcamStarted) {
        setStatus('Model loaded, but Webcam failed. Check permissions.', 'error');
    }

    updateButtonState();
    console.log("Initialization sequence complete.");
}

if (!window.indexedDB) {
     setStatus("Error: IndexedDB is not supported by this browser. Caching disabled.", 'error');
     predictButton.disabled = true; buttonText.textContent = 'Browser Incompatible';
     loadTimeValueSpan.textContent = 'N/A'; latencyValueSpan.textContent = 'N/A';
} else if (typeof ort === 'undefined') {
     setStatus("Error: ONNX Runtime script not loaded. Refresh maybe?", 'error');
     predictButton.disabled = true; buttonText.textContent = 'ORT Load Failed';
     loadTimeValueSpan.textContent = 'N/A'; latencyValueSpan.textContent = 'N/A';
} else if (typeof DecompressionStream === 'undefined') {
     setStatus("Error: Browser lacks DecompressionStream API. Cannot run.", 'error');
     predictButton.disabled = true; buttonText.textContent = 'Browser Incompatible';
     loadTimeValueSpan.textContent = 'N/A'; latencyValueSpan.textContent = 'N/A';
} else {
    initializeApp();
}

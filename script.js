import { client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const sourcePreview = document.getElementById('source-preview');
    const removeBtn = document.getElementById('remove-btn');
    
    const modelSelect = document.getElementById('model-select');
    const modelDesc = document.getElementById('model-desc');
    const upscaleBtn = document.getElementById('upscale-btn');
    
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    
    const downloadBtn = document.getElementById('download-btn');
    const sliderContainer = document.getElementById('slider-container');
    const sliderBefore = document.getElementById('slider-before');
    const sliderAfter = document.getElementById('slider-after');
    const sliderHandle = document.getElementById('slider-handle');

    // Global App State
    let selectedFile = null;
    let isDragging = false;
    let upscaledObjectUrl = null;
    let originalObjectUrl = null;
    let gradioApp = null;
    let fallbackApp = null;

    async function getGradioApp(useFallback = false) {
        if (useFallback) {
            if (!fallbackApp) {
                console.log("Connecting to fallback cloud space (akhaliq)...");
                fallbackApp = await client("akhaliq/CodeFormer");
            }
            return fallbackApp;
        } else {
            if (!gradioApp) {
                console.log("Connecting to main cloud space (sczhou)...");
                gradioApp = await client("sczhou/CodeFormer");
            }
            return gradioApp;
        }
    }

    // Client-side image compression / downscaling helper to prevent GPU timeouts & HF rate limits
    async function preprocessImage(file, maxDimension = 1600) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                let width = img.width;
                let height = img.height;
                
                if (width <= maxDimension && height <= maxDimension) {
                    resolve(file);
                    return;
                }
                
                if (width > height) {
                    if (width > maxDimension) {
                        height = Math.round((height * maxDimension) / width);
                        width = maxDimension;
                    }
                } else {
                    if (height > maxDimension) {
                        width = Math.round((width * maxDimension) / height);
                        height = maxDimension;
                    }
                }
                
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                
                canvas.toBlob((blob) => {
                    const originalName = file.name.substring(0, file.name.lastIndexOf('.'));
                    const processedFile = new File([blob], `${originalName}_preprocessed.jpg`, {
                        type: 'image/jpeg',
                        lastModified: Date.now()
                    });
                    resolve(processedFile);
                }, 'image/jpeg', 0.92);
            };
            
            img.onerror = () => {
                resolve(file);
            };
            
            img.src = URL.createObjectURL(file);
        });
    }

    const modelDescriptions = {
        'codeformer-ultra-4x': 'Uses cloud GPUs to restore faces (CodeFormer) and upscale the background (Real-ESRGAN) to 4K resolution. Ideal for photos, portraits, and general scenes.',
        'realesrgan-clean-4x': 'Uses cloud GPUs running Real-ESRGAN to upscale images 4x. Perfect for high-quality digital art, anime, graphics, and noise-free illustrations.'
    };

    // Update description when model changes
    modelSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        modelDesc.textContent = modelDescriptions[val] || '';
    });

    // Setup Drag and Drop / File triggers
    dropzone.addEventListener('click', (e) => {
        // Prevent trigger if clicking the remove button
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, WEBP).');
            return;
        }

        selectedFile = file;

        // Clean previous object URLs to prevent leaks
        if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
        
        // Preview selected file
        originalObjectUrl = URL.createObjectURL(file);
        sourcePreview.src = originalObjectUrl;
        
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        upscaleBtn.disabled = false;
        
        // Hide results if we select a new image
        resultsSection.classList.add('hidden');
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        
        if (originalObjectUrl) {
            URL.revokeObjectURL(originalObjectUrl);
            originalObjectUrl = null;
        }
        if (upscaledObjectUrl) {
            URL.revokeObjectURL(upscaledObjectUrl);
            upscaledObjectUrl = null;
        }

        sourcePreview.src = '';
        uploadPrompt.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        upscaleBtn.disabled = true;
        
        resultsSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
    }

    // Submit Upscale Request
    upscaleBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Feedback - Processing State
        upscaleBtn.disabled = true;
        modelSelect.disabled = true;
        removeBtn.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            // Setup specific model parameters based on user selection
            const modelId = modelSelect.value;
            const faceUpsample = (modelId === "codeformer-ultra-4x");
            const faceAlign = faceUpsample;

            // Preprocess/downscale image client-side to prevent timeouts and HF rate limits
            console.log("Preprocessing image client-side...");
            const processedFile = await preprocessImage(selectedFile, 1600);

            let app;
            let result;
            try {
                app = await getGradioApp(false);
                console.log(`Calling primary cloud GPU API for ${modelId}...`);
                result = await app.predict("/inference", [
                    processedFile,       // image (File object)
                    faceAlign,          // face_align (boolean)
                    true,               // background_enhance (boolean)
                    faceUpsample,       // face_upsample (boolean)
                    4.0,                // upscale factor (float)
                    0.6                 // codeformer_fidelity (float)
                ]);
            } catch (firstError) {
                console.warn("Primary space failed or busy, trying fallback space...", firstError);
                app = await getGradioApp(true);
                console.log(`Calling fallback cloud GPU API for ${modelId}...`);
                result = await app.predict("/inference", [
                    processedFile,       // image (File object)
                    faceAlign,          // face_align (boolean)
                    true,               // background_enhance (boolean)
                    faceUpsample,       // face_upsample (boolean)
                    4.0,                // upscale factor (float)
                    0.6                 // codeformer_fidelity (float)
                ]);
            }

            const outputImg = result.data[0];
            if (!outputImg || (!outputImg.url && !outputImg.data)) {
                throw new Error("Invalid output received from cloud service.");
            }

            // Resolve URL (handles absolute, relative, or base64 data)
            let imageUrl = outputImg.url;
            if (!imageUrl && outputImg.data) {
                imageUrl = outputImg.data;
            }
            if (imageUrl && !imageUrl.startsWith('http') && !imageUrl.startsWith('data:')) {
                imageUrl = `https://sczhou-codeformer.hf.space${imageUrl.startsWith('/') ? '' : '/'}${imageUrl}`;
            }

            // Fetch the image to convert it into a local object URL to bypass CORS during download
            const imageResponse = await fetch(imageUrl);
            const imageBlob = await imageResponse.blob();
            
            // Cleanup previous result url if any
            if (upscaledObjectUrl) URL.revokeObjectURL(upscaledObjectUrl);
            
            upscaledObjectUrl = URL.createObjectURL(imageBlob);

            // Populate images into comparison slider
            sliderBefore.src = originalObjectUrl;
            sliderAfter.src = upscaledObjectUrl;
            
            // Set download details
            downloadBtn.href = upscaledObjectUrl;
            const originalName = selectedFile.name.substring(0, selectedFile.name.lastIndexOf('.'));
            downloadBtn.download = `${originalName}_upscaled.png`;

            // Reset slider position to center
            sliderContainer.style.setProperty('--slide-pos', '50%');

            // Show results
            resultsSection.classList.remove('hidden');
            
            // Smoothly scroll down to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);

        } catch (error) {
            console.error('Error during upscale:', error);
            alert(`An error occurred: ${error.message}\nIf this is the first time running the model, the Hugging Face space might be waking up. Please try again in a few seconds.`);
        } finally {
            upscaleBtn.disabled = false;
            modelSelect.disabled = false;
            removeBtn.classList.remove('hidden');
            loadingSection.classList.add('hidden');
        }
    });

    // Comparison Slider Functionality (Supports Mouse and Touch Events)
    const startDrag = (e) => {
        isDragging = true;
        e.preventDefault();
    };

    const stopDrag = () => {
        isDragging = false;
    };

    const drag = (e) => {
        if (!isDragging) return;
        
        // Prevent touch scrolling during drag
        if (e.cancelable) {
            e.preventDefault();
        }
        
        // Support mouse and touch coordinate parsing
        let clientX;
        if (e.touches) {
            clientX = e.touches[0].clientX;
        } else {
            clientX = e.clientX;
        }

        const rect = sliderContainer.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        
        let positionPercent = (offsetX / rect.width) * 100;
        
        // Constrain percentage between 0% and 100%
        positionPercent = Math.max(0, Math.min(100, positionPercent));
        
        // Update CSS variable --slide-pos
        sliderContainer.style.setProperty('--slide-pos', `${positionPercent}%`);
    };

    // Attach slider handlers
    sliderHandle.addEventListener('mousedown', startDrag);
    window.addEventListener('mouseup', stopDrag);
    window.addEventListener('mousemove', drag);

    sliderHandle.addEventListener('touchstart', startDrag);
    window.addEventListener('touchend', stopDrag);
    window.addEventListener('touchmove', drag);

    // Support clicking anywhere on the slider to move handle
    sliderContainer.addEventListener('click', (e) => {
        // Prevent click trigger if dragging
        if (isDragging) return;
        
        const rect = sliderContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        let positionPercent = (offsetX / rect.width) * 100;
        positionPercent = Math.max(0, Math.min(100, positionPercent));
        sliderContainer.style.setProperty('--slide-pos', `${positionPercent}%`);
    });
});

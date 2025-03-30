document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileLabel = document.getElementById('file-label');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const resultContainer = document.getElementById('result-container');
    const resultImage = document.getElementById('result-img');
    const resultClass = document.getElementById('result-class');
    const resultConfidence = document.getElementById('result-confidence');
    const progressFill = document.getElementById('progress-fill');
    const tryAgainButton = document.getElementById('try-again-button');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        updateFileInfo();
    }

    // Handle file input change
    fileInput.addEventListener('change', updateFileInfo);

    function updateFileInfo() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileName.textContent = file.name;
            fileLabel.textContent = 'File selected';
            uploadButton.removeAttribute('disabled');
        }
    }

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('Please select a file first');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
                return;
            }

            // Update results
            resultImage.src = result.image_path;
            resultClass.textContent = result.class.charAt(0).toUpperCase() + result.class.slice(1);
            resultConfidence.textContent = result.confidence.toFixed(2) + '%';
            progressFill.style.width = result.confidence + '%';

            // Show results
            resultContainer.style.display = 'block';
            dropArea.style.display = 'none';

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request');
        } finally {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        }
    });

    // Try again button
    tryAgainButton.addEventListener('click', function() {
        resultContainer.style.display = 'none';
        dropArea.style.display = 'block';
        fileInput.value = '';
        fileName.textContent = '';
        fileLabel.textContent = 'Drag & Drop or Click to Upload';
    });
});
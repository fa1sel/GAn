document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const chatHistory = document.getElementById('chat-history');
    const promptInput = document.getElementById('prompt-input');
    const sendButton = document.getElementById('send-button');
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('temp-value');
    const numSamplesInput = document.getElementById('numSamples');
    const maxLengthInput = document.getElementById('maxLength'); // Added maxLength input

    // --- Event Listeners ---

    // Update temperature display value
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Handle sending message on button click
    sendButton.addEventListener('click', sendMessage);

    // Handle sending message on Enter key press in textarea (Shift+Enter for newline)
    promptInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default newline behavior
            sendMessage();
        }
    });

    // Auto-resize textarea
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto'; // Reset height
        promptInput.style.height = (promptInput.scrollHeight) + 'px'; // Set to scroll height
    });


    // --- Core Functions ---

    /**
     * Adds a message to the chat history UI.
     * @param {string} text - The message text.
     * @param {string} sender - 'user' or 'bot'.
     */
    function addMessageToChat(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

        const paragraph = document.createElement('p');
        paragraph.textContent = text; // Use textContent for security
        messageDiv.appendChild(paragraph);

        chatHistory.appendChild(messageDiv);
        // Scroll to the bottom of the chat history
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    /**
     * Shows a loading indicator in the chat.
     */
    function showLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('loading-indicator');
        loadingDiv.id = 'loading-indicator'; // Assign ID to remove later
        loadingDiv.innerHTML = `<span>Generating</span><div class="dot-flashing"></div>`;
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    /**
     * Removes the loading indicator from the chat.
     */
    function hideLoadingIndicator() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    /**
     * Handles sending the prompt to the backend and displaying the response.
     */
    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return; // Don't send empty messages

        // Get parameter values
        const temperature = parseFloat(tempSlider.value);
        const numSamples = parseInt(numSamplesInput.value);
        const maxLength = parseInt(maxLengthInput.value); // Get max length

        // Display user message immediately
        addMessageToChat(prompt, 'user');

        // Clear input field and disable controls
        promptInput.value = '';
        promptInput.style.height = 'auto'; // Reset height after clearing
        promptInput.disabled = true;
        sendButton.disabled = true;
        showLoadingIndicator();

        try {
            // --- API Call to Backend ---
            // Replace '/generate' with your actual backend endpoint URL
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt, // Send the user's prompt
                    temperature: temperature,
                    numSamples: numSamples,
                    maxLength: maxLength // Send max length
                    // Add any other parameters your backend expects
                }),
            });

            hideLoadingIndicator(); // Hide loading indicator regardless of success/failure

            if (!response.ok) {
                // Try to get error message from backend response
                let errorMsg = `Request failed with status ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(errorMsg);
            }

            const result = await response.json();

            // Display bot response(s)
            if (result.generated_texts && result.generated_texts.length > 0) {
                 // If multiple samples, display each in a separate bubble
                 result.generated_texts.forEach((text, index) => {
                    const sampleHeader = numSamples > 1 ? `Sample ${index + 1}:\n` : '';
                    addMessageToChat(sampleHeader + text, 'bot');
                 });
            } else {
                addMessageToChat("Sorry, I couldn't generate a response.", 'bot');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            hideLoadingIndicator(); // Ensure loading is hidden on error
            addMessageToChat(`Error: ${error.message}`, 'bot'); // Display error in chat
        } finally {
            // Re-enable input controls
            promptInput.disabled = false;
            sendButton.disabled = false;
            promptInput.focus(); // Focus back on the input field
        }
    }
});

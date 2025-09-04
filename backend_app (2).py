# backend_app.py
import os
import json
import traceback # For detailed error logging
import gc # Garbage collector

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize # Assuming NLTK is used for tokenization

# --- Configuration (MUST MATCH TRAINING/MODEL) ---
PROJECT_PATH = 'C:/Users/PMLS/Desktop/FYP Main Project/FYP Main Project/GAN-Output-Large' # Directory containing model/vocab
VOCAB_PATH = os.path.join(PROJECT_PATH, 'vocab_large.json')
MODEL_PATH = os.path.join(PROJECT_PATH, 'trained_model.pt')

# Model hyperparameters (MUST match the saved model's architecture)
embedding_dim = 256
hidden_dim = 512
num_layers = 2 # LSTM layers
dropout = 0.2
# vocab_size will be loaded from the vocab file

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Backend using device: {DEVICE} ---")

# --- Model Definition (Generator Class) ---
# Include the exact Generator class definition used for training here
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        lstm_output_dim = hidden_dim * 2 # Bidirectional
        self.attention = nn.MultiheadAttention(lstm_output_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(lstm_output_dim)
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim * 2)
        self.gelu = nn.GELU()
        self.dropout_ff = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_output_dim * 2, lstm_output_dim)
        self.layer_norm2 = nn.LayerNorm(lstm_output_dim)
        self.output_layer = nn.Linear(lstm_output_dim, vocab_size)
        self.dropout_emb = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout_emb(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out1 = self.layer_norm1(lstm_out + attn_output)
        residual = out1
        out2 = self.fc1(out1)
        out2 = self.gelu(out2)
        out2 = self.dropout_ff(out2)
        out2 = self.fc2(out2)
        out2 = self.layer_norm2(out2 + residual)
        logits = self.output_layer(out2)
        return logits

# --- Utility Functions ---

def load_vocab(path):
    """Loads vocabulary mappings from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        id2word_loaded = {int(k): v for k, v in vocab_data['id2word'].items()}
        word2id_loaded = vocab_data['word2id']
        vocab_size = len(word2id_loaded)
        print(f"Vocabulary loaded from {path}, size: {vocab_size}")
        # Basic validation
        if '<PAD>' not in word2id_loaded or '<UNK>' not in word2id_loaded or \
           '<START>' not in word2id_loaded or '<END>' not in word2id_loaded:
            print("Warning: Standard special tokens missing from loaded vocabulary!")
        return word2id_loaded, id2word_loaded, vocab_size
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found at {path}")
        return None, None, 0
    except Exception as e:
        print(f"ERROR: Could not load vocabulary: {e}")
        return None, None, 0

def load_generator_for_inference(model_path, vocab_size, config, device):
    """Loads the trained generator model state."""
    # Instantiate the model with parameters from config
    generator = Generator(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    try:
        # Load the saved state dictionary
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state"])

        #generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval() # Set to evaluation mode IMPORTANT!
        print(f"Generator loaded successfully from {model_path}")
        print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        return generator
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load generator state dict: {e}")
        print(traceback.format_exc())
        return None

def generate_text(generator, word2id, id2word, seq_length, num_samples, temperature, device, prompt=None):
    """Generates text samples using the loaded generator."""
    if not generator or not word2id or not id2word:
        print("ERROR: Generator or vocabulary not available for generation.")
        return ["Error: Model/Vocab not loaded."]

    generator.eval() # Ensure model is in eval mode

    generated_texts = []
    start_token_id = word2id.get('<START>', -1) # Use -1 to check if found
    end_token_id = word2id.get('<END>', -1)
    unk_id = word2id.get('<UNK>', -1)
    pad_id = word2id.get('<PAD>', -1)

    # Validate special tokens
    if -1 in [start_token_id, end_token_id, unk_id, pad_id]:
        print("ERROR: One or more special tokens (<START>, <END>, <UNK>, <PAD>) not found in word2id.")
        return ["Error: Vocabulary integrity issue."]

    special_ids_to_skip = {start_token_id, end_token_id, pad_id}

    print(f"Generating {num_samples} samples (temp={temperature}, max_len={seq_length})...")
    if prompt: print(f"Using prompt: '{prompt}'")

    with torch.no_grad():
        for i in range(num_samples):
            # --- Tokenize Prompt ---
            if prompt:
                try:
                    # Use NLTK's word_tokenize (ensure 'punkt' is downloaded)
                    #prompt_tokens = word_tokenize(prompt)
                    prompt_tokens = prompt.strip().replace('،', '').split()

                except Exception as e:
                    print(f"Warning: NLTK tokenization failed for prompt '{prompt}': {e}")
                    prompt_tokens = prompt.split() # Fallback to simple split

                current_sequence_ids = [word2id.get(token, unk_id) for token in prompt_tokens]
                if not current_sequence_ids:
                    print("Warning: Prompt resulted in empty token sequence, starting with <START>.")
                    current_sequence_ids = [start_token_id]
            else:
                current_sequence_ids = [start_token_id]

            generated_ids = list(current_sequence_ids) # Store the full sequence

            # --- Generation Loop ---
            for _ in range(seq_length):
                # Prepare input tensor (use tail end as context)
                # The context length the model can handle depends on training seq_length
                # Using a fixed context window size might be more robust if prompts vary wildly
                context_window = 50 # Example: Use last 50 tokens as context
                input_ids_context = current_sequence_ids[-context_window:]
                input_tensor = torch.LongTensor([input_ids_context]).to(device)

                try:
                    output_logits = generator(input_tensor)
                except Exception as e:
                    print(f"ERROR during model forward pass: {e}")
                    print(traceback.format_exc())
                    generated_texts.append(f"[Model Error: {e}]")
                    break # Stop generation for this sample on error

                # Get logits for the next token
                next_token_logits = output_logits[0, -1, :]

                # Apply temperature
                scaled_logits = next_token_logits / max(temperature, 1e-8) # Avoid division by zero

                # Get probabilities
                next_token_probs = F.softmax(scaled_logits, dim=-1)

                # Sample next token
                try:
                  next_token_id = torch.multinomial(next_token_probs, 1).item()
                except RuntimeError as e:
                  print(f"Error during multinomial sampling (check probabilities): {e}")
                  print(f"Probabilities sum: {next_token_probs.sum()}")
                  print(f"Probabilities has NaN: {torch.isnan(next_token_probs).any()}")
                  # Fallback: use argmax or break
                  next_token_id = torch.argmax(next_token_probs).item()
                  # break # Or stop generation for this sample

                # Stop if <END> token
                if next_token_id == end_token_id:
                    break

                generated_ids.append(next_token_id)
                current_sequence_ids.append(next_token_id)

            # --- Decode Sequence ---
            output_words = [id2word.get(str(gid), "<UNK>") # Ensure gid is string for dict lookup if keys were saved as str
                            for gid in generated_ids if gid not in special_ids_to_skip]

            # Handle cases where only prompt was present or nothing generated
            #prompt_word_count = len(word_tokenize(prompt)) if prompt else 0
            prompt_word_count = len(prompt_tokens) if prompt else 0

            if len(output_words) > prompt_word_count:
                generated_text = ' '.join(output_words)
                generated_texts.append(generated_text)
                print(f"  Sample {i+1} generated.")
            elif not generated_texts and i == num_samples - 1: # If no samples generated at all
                 generated_texts.append("(No text generated beyond prompt)")


    return generated_texts


# --- Roman Urdu Conversion Placeholder ---
def convert_roman_to_urdu(text):
    """
    Placeholder function for Roman Urdu to Urdu conversion.
    Replace this with your actual implementation using a library like urduhack
    or a custom transliteration model/ruleset.
    """
    print(f"Attempting Roman Urdu conversion for: '{text}' (Placeholder)")
    # Example (very basic, needs proper library):
    # if text.lower() == "salam":
    #     return "سلام"
    # if text.lower() == "pakistan":
    #     return "پاکستان"

    # For now, assume input is already Urdu or return as is
    print("Placeholder: Returning input as is.")
    return text
# -----------------------------------------


# --- Load Model and Vocabulary ONCE at Startup ---
print("--- Initializing Backend: Loading Resources ---")
word2id, id2word, vocab_size = load_vocab(VOCAB_PATH)

# Prepare config dict for model loading
model_config = {
    'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim,
    'num_layers': num_layers, 'dropout': dropout,
}

generator_model = None
if vocab_size > 0:
    generator_model = load_generator_for_inference(MODEL_PATH, vocab_size, model_config, DEVICE)
else:
    print("ERROR: Vocabulary loading failed. Model cannot be loaded.")

# Clean up memory after loading
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"GPU Memory used after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print("--- Resource Loading Complete ---")
# ----------------------------------here ---------------

import random  # at the top with other imports if not already imported

def sample_from_logits(logits, temperature=1.0):
    """Sample a token id from logits with temperature."""
    scaled_logits = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled_logits, dim=-1)
    next_token_id = torch.multinomial(probs, 1).item()
    return next_token_id

def decode_tokens(token_ids, id2word):
    """Convert list of token ids to string text using id2word dict."""
    words = [id2word.get(str(tid), "<UNK>") for tid in token_ids]
    return ' '.join(words)

def generate_unconditional(model, word2id, id2word, max_len=50, temperature=1.0, device='cpu'):
    model.eval()
    generated_texts = []

    with torch.no_grad():
        current_sequence = [word2id['<START>']]

        for _ in range(max_len):
            input_tensor = torch.LongTensor([current_sequence]).to(device)
            output = model(input_tensor)

            next_token_logits = output[0, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=0)
            next_token_id = torch.multinomial(next_token_probs, 1).item()

            current_sequence.append(next_token_id)

            if next_token_id == word2id['<END>']:
                break

       
        output_words = [
         id2word[token_id] if token_id in id2word else '<UNK>'
         for token_id in current_sequence
         if token_id not in {word2id['<START>'], word2id['<END>'], word2id['<PAD>']}
   ]






        print(f"<START>: {word2id['<START>']}, <END>: {word2id['<END>']}, <PAD>: {word2id['<PAD>']}")
        print("Generated token IDs:", current_sequence)
        print("Next token logits:", next_token_logits[:10])  # Show first 10 values
        print("Sampled token ID:", next_token_id)

        generated_text = ' '.join(output_words)
        return generated_text



# --- Create Flask App ---
app = Flask(__name__) # Looks for templates/static folders relative to this file

#@app.route('/')
#def index():
    #return "Server is running! Use POST /generate to get text generation."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def handle_generation():
    print("\n--- Received /generate request ---")

    if generator_model is None or vocab_size == 0:
        print("ERROR: Model or vocabulary not loaded.")
        return jsonify({"error": "Model is not available on the server."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON data received."}), 400

        prompt_input = data.get('prompt', '')
        temp = float(data.get('temperature', 0.8))
        max_len = int(data.get('maxLength', 50))
        num_samples = int(data.get('numSamples', 1))

        # THIS LINE FORCES unconditional generation (ignore client input)
        use_prompt_for_generation = False  

        # If you want to respect client input, use this instead:
        # use_prompt_for_generation = bool(data.get('usePrompt', True))

        if not (0.1 <= temp <= 2.0): temp = 0.8
        if not (10 <= max_len <= 300): max_len = 50
        if not (1 <= num_samples <= 5): num_samples = 1

        print(f"Request Params: Temp={temp}, Samples={num_samples}, MaxLen={max_len}")
        print(f"Input Prompt: '{prompt_input}'")
        print(f"Use prompt for generation? {use_prompt_for_generation}")

    except Exception as e:
        print(f"Error parsing request data: {e}")
        return jsonify({"error": f"Invalid request data format: {e}"}), 400

    # Roman Urdu conversion (optional)
    try:
        final_prompt = convert_roman_to_urdu(prompt_input)
        if final_prompt != prompt_input:
            print(f"Prompt after conversion (placeholder): '{final_prompt}'")
        else:
            print("No conversion applied (placeholder).")
    except Exception as e:
        print(f"Error during Roman Urdu conversion (placeholder): {e}")
        final_prompt = prompt_input

    generated_samples = []
    try:
        if use_prompt_for_generation:
            # Use your existing generate_text function (prompt-based)
            generated_samples = generate_text(
                generator=generator_model,
                word2id=word2id,
                id2word=id2word,
                seq_length=max_len,
                num_samples=num_samples,
                temperature=temp,
                device=DEVICE,
                prompt=final_prompt
            )
        else:
            # Generate unconditional samples ignoring prompt tokens
            for _ in range(num_samples):
                gen_text = generate_unconditional(
                    model=generator_model,
                    word2id=word2id,
                    id2word=id2word,
                    max_len=max_len,
                    temperature=temp,
                    device=DEVICE
                )
                generated_samples.append(gen_text)

        print(f"Generated {len(generated_samples)} samples.")
        return jsonify({
            "prompt": final_prompt,   # Show prompt text but unused in generation
            "generated_texts": generated_samples
        })

    except Exception as e:
        print(f"ERROR during text generation: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Server error during text generation."}), 500
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    print("Open your browser and go to: http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False)

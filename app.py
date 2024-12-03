%%writefile app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import streamlit as st

# 1. Define the MinimalTransformer model
class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer for character input
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))  # Positional encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)  # Output layer to predict next character

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)  # Positional encoding for each input
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through the transformer encoder
        x = self.output_layer(x)  # Output layer to get predictions
        return x


# 2. Define the NameDataset class
class NameDataset(Dataset):
    def __init__(self, txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.names = [line.strip() for line in f]
        self.chars = sorted(list(set(''.join(self.names)))) + [' ']
        self.vocab_size = len(self.chars)
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)


# Function to generate a name from the trained model
def generate_name(model, dataset, start_chars, max_length=30, temperature=1.0):
    model.eval()
    start_chars = start_chars.capitalize()  # Ensure only the first character is uppercase
    input_tensor = torch.tensor([dataset.char_to_int[char] for char in start_chars], dtype=torch.long).unsqueeze(0)
    generated_name = start_chars

    with torch.no_grad():
        for _ in range(max_length - len(start_chars)):
            output = model(input_tensor)
            logits = output[0, -1, :] / temperature  # Adjust temperature to control randomness
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()

            # Sample from the probabilities to choose the next character
            predicted_char_idx = torch.multinomial(torch.tensor(probabilities), num_samples=1).item()
            predicted_char = dataset.int_to_char[predicted_char_idx]

            if predicted_char == ' ':
                break
            generated_name += predicted_char
            input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_char_idx]], dtype=torch.long)], dim=1)

    return generated_name


# Streamlit interface
st.title("Lietuviškų vardų generatorius")

# User selects name type
name_type = st.selectbox("Pasirinkite vardo tipą", ["Vyriškas", "Moteriškas"])

# Load the correct dataset and model
if not os.path.exists('model_male.pth') or not os.path.exists('model_female.pth'):
    st.error("Model files are missing. Please ensure 'model_male.pth' and 'model_female.pth' are in the correct directory.")
else:
    if name_type == "Vyriškas":
        dataset = NameDataset('vardai_male.txt')
        model = MinimalTransformer(vocab_size=dataset.vocab_size, embed_size=128, num_heads=8, forward_expansion=4)
        model.load_state_dict(torch.load('model_male.pth'), strict=False)
    else:
        dataset = NameDataset('vardai_female.txt')
        model = MinimalTransformer(vocab_size=dataset.vocab_size, embed_size=128, num_heads=8, forward_expansion=4)
        model.load_state_dict(torch.load('model_female.pth'), strict=False)

    # Input field for the starting letter(s)
    start_chars = st.text_input("Įveskite pradines raides", "")

    # Slider for temperature

    temperature = st.slider("Pasirinkite atsitiktinumo lygį (temperatūra)", 0.5, 2.0, 1.0)

    # Button to generate a name
    if st.button("Generuoti vardą"):
        if start_chars:
            # Generate and display the name
            name = generate_name(model, dataset, start_chars, temperature=temperature)
            st.write(f"Sugeneruotas Vardas: {name}")
        else:
            st.warning("Prašome įvesti pradines raides!")

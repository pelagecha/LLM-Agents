from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
print("Loading the model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Define the prompt
print("Input your prompt:")
prompt = f"{str(input())}\n"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.1)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
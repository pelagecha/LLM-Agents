import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the retriever models and tokenizers
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

print("Loading the retriever models...")
question_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

# Step 2: Load the generator model and tokenizer
print("Loading the generator model...")
generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
generator_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)

# Add padding token if not present
generator_tokenizer.pad_token = generator_tokenizer.eos_token

# Step 3: Create a document store (in a real RAG system, this might be a database or a more sophisticated retrieval system)
documents = [
    "Give single-word answers. Don't include Explanation",
]

# Step 4: Encode documents with context encoder
print("Encoding documents...")
def encode_documents(documents):
    inputs = context_tokenizer(documents, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = context_encoder(**inputs).pooler_output
    return embeddings

document_embeddings = encode_documents(documents)

# Step 5: Define a function to retrieve relevant documents based on input questions
def retrieve_relevant_documents(question, top_k=3, threshold=0.8):
    # Encode the question
    inputs = question_tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output

    # Compute similarity scores
    similarities = cosine_similarity(question_embedding.cpu().numpy(), document_embeddings.cpu().numpy())
    relevant_indices = np.argsort(similarities[0])[::-1][:top_k]

    # Filter documents by relevance threshold
    filtered_docs = [
        documents[i] for i in relevant_indices if similarities[0][i] >= threshold
    ]
    return filtered_docs

def main(inp):
    # Step 6: Define the RAG pipeline
    question = inp

    # Retrieve relevant documents
    retrieved_docs = retrieve_relevant_documents(question, top_k=3, threshold=0.8)
    retrieved_context = " ".join(retrieved_docs)

    # Generate response based on the retrieved context and question
    prompt = f"{retrieved_context}\n\nThe following is an answer to the given question based on the provided context.\nQuestion: {question}\nAnswer:"
    inputs = generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding='max_length').to(device)

    try:
        with torch.no_grad():
            outputs = generator_model.generate(
                **inputs,
                max_new_tokens=150,  # Increased max_new_tokens to prevent cutoff
                num_return_sequences=1,  # Generate one response
                do_sample=True,
                top_p=0.9,
                temperature=0.3
            )
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response"

    # Decode the response
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 7: Read questions from file and generate answers
print("Processing riddles...")
with open("riddles.txt", "r") as f:
    count = 0
    for line in f:
        if count > 5:
            break
        if "<SPLIT>" in line:
            q, a = line.split("<SPLIT>")
            generated_answer = main(q.strip())
            print("=====================================================")
            print(f"Generated Answer: {generated_answer}\nCorrect Answer: {a.strip()}\n")
            count += 1
        else:
            print("Line missing <SPLIT> delimiter, skipping.")

# Clear cache to manage GPU memory
torch.cuda.empty_cache()
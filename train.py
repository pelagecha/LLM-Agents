import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the retriever models and tokenizers
logging.set_verbosity_error()
print("Loading the retriever models...")
question_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Step 2: Load the generator model and tokenizer
print("Loading the generator model...")
generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
generator_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Step 3: Create a document store (in a real RAG system, this might be a database or a more sophisticated retrieval system)
documents = [
    "Give single-word answers. Don't include Explanation",
]

# Step 4: Encode documents with context encoder
def encode_documents(documents):
    doc_embeddings = []
    for doc in documents:
        inputs = context_tokenizer(doc, return_tensors="pt")
        with torch.no_grad():
            embedding = context_encoder(**inputs).pooler_output
        doc_embeddings.append(embedding)
    return torch.cat(doc_embeddings, dim=0)

document_embeddings = encode_documents(documents)

# Step 5: Define a function to retrieve relevant documents based on input questions
def retrieve_relevant_documents(question, top_k=1):
    # Encode the question
    inputs = question_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output
    
    # Compute similarity scores
    similarities = cosine_similarity(question_embedding.numpy(), document_embeddings.numpy())
    relevant_indices = np.argsort(similarities[0])[::-1][:top_k]
    return [documents[i] for i in relevant_indices]

def main(inp):
    # Step 6: Define the RAG pipeline
    # print("Input your question:")
    question = inp

    # Retrieve relevant documents
    retrieved_docs = retrieve_relevant_documents(question, top_k=1)
    retrieved_context = " ".join(retrieved_docs)

    # Generate response based on the retrieved context and question
    prompt = f"\nQuestion: {question}\nAnswer:"
    inputs = generator_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = generator_model.generate(**inputs, max_length=50, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.3)

    # Decode and print the response
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    return response


with open("riddles.txt", "r") as f:
    count = 0
    for line in f.readlines():
        if count > 5:
            break
        q, a = line.split("<SPLIT>")
        print("=====================================================\n")
        print(f"{main(q)}\n >> Correct Answer:{a}")
        count += 1
        

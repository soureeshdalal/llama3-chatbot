<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1 align="left">Fine-Tuned Llama 3.2 Chatbot</h1>
<p align="left">
    A powerful chatbot fine-tuned on <strong>Llama 3.2</strong>, enhanced with <strong>Hybrid RAG (Retrieval-Augmented Generation)</strong> and optimized for advanced code generation, debugging, and explanation.
</p>

---

<h2> Overview</h2>
<p>
This project fine-tunes the <strong>Llama 3.2</strong> model for enhanced reasoning, programming assistance, and contextual awareness. Unlike standard fine-tuning, this approach integrates **Hybrid RAG (BM25 + ChromaDB embeddings)** for a more dynamic and knowledge-enhanced chatbot experience.
</p>

<h3> Standout Features:</h3>
<ul>
    <li> <strong>Fine-Tuned on Python Code Datasets</strong> - Trained on **CodeAlpaca-20k** and other Python-centric datasets.</li>
    <li><strong>Hybrid RAG Implementation</strong> - Combines **BM25 Sparse Retrieval** with **Dense ChromaDB Embeddings**.</li>
    <li> <strong>Optimized for Limited Compute</strong> - JAX-based fine-tuning with **Colab T4 GPU constraints in mind**.</li>
    <li><strong>Context-Aware Generation</strong> - RAG-enhanced responses for **complex Python questions**.</li>
    <li><strong>Cloud Deployment</strong> - Hosted on **Hugging Face Spaces** for easy access and inference.</li>
</ul>

<h3>Model Details:</h3>
<ul>
    <li><strong>Base Model:</strong> Llama 3.2</li>
    <li><strong>Fine-Tuning Method:</strong> Full Model Fine-Tuning (JAX-based approach)</li>
    <li><strong>VectorDB:</strong> <a href="https://github.com/chroma-core/chroma">ChromaDB</a> + BM25 Hybrid Search</li>
    <li><strong>Deployment:</strong> Hugging Face Spaces</li>
    <li><strong>Use Cases:</strong> Code Generation, Debugging, Code Explanation, and RAG-based Queries</li>
</ul>

---

<h2>Model & RAG Files</h2>
<p>The following files are included in the repository:</p>
<table>
    <tr>
        <th>File</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>config.json</code></td>
        <td>Model configuration file</td>
    </tr>
    <tr>
        <td><code>tokenizer.json</code></td>
        <td>Tokenizer settings and vocabulary</td>
    </tr>
    <tr>
        <td><code>model.safetensors</code></td>
        <td>Fine-tuned model weights</td>
    </tr>
    <tr>
        <td><code>chroma_db/</code></td>
        <td>VectorDB files storing RAG embeddings</td>
    </tr>
    <tr>
        <td><code>README.md</code></td>
        <td>Project documentation</td>
    </tr>
</table>

---

<h2>Installation & Setup</h2>
<h3>1. Install Dependencies</h3>
<p>Ensure you have the required packages installed:</p>
<pre>
pip install transformers torch gradio chromadb huggingface_hub safetensors faiss-cpu
</pre>

<h3>2. Load the Fine-Tuned Model</h3>
<p>Use the following script to load the **Llama 3.2** fine-tuned model:</p>

<pre>
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_repo = "soureesh1211/finetuned-llama3"
model = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_repo)
</pre>

<h3>3. Run Inference</h3>
<pre>
input_text = "How do I create a class in Python?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)
</pre>

---

<h2>Running the Chatbot</h2>
<p>A <strong>Gradio-based chatbot</strong> has been deployed for easy interaction with the fine-tuned model.</p>

<h3>1. Clone the Repository</h3>
<pre>
git clone https://huggingface.co/spaces/soureesh1211/finetuned-llama3-chatbot
cd finetuned-llama3-chatbot
</pre>

<h3>2. Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>3. Run the Chatbot</h3>
<pre>
python app.py
</pre>

---

<h2>Hugging Face Model & Space</h2>
<ul>
    <li><strong>Fine-Tuned Model:</strong> <a href="https://huggingface.co/soureesh1211/finetuned-llama3">soureesh1211/finetuned-llama3</a></li>
    <li><strong>Live Chatbot:</strong> <a href="https://huggingface.co/spaces/soureesh1211/finetuned-llama3-chatbot">Hugging Face Space</a></li>
</ul>

---

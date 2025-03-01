<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

</head>
<body>

<h1 align="left">🚀 Fine-Tuned Llama 3.2 Chatbot</h1>
<p align="left">
    A powerful chatbot fine-tuned on <strong>Llama 3.2</strong>, optimized for advanced code generation, debugging, and explanation.
</p>

---

<h2>Overview</h2>
<p>
This project fine-tunes the <strong>Llama 3.2</strong> model for enhanced reasoning, programming assistance, and contextual awareness. The fine-tuning process leverages a comprehensive dataset of Python code to improve the model's performance in code-related tasks.
</p>

<h3> Features:</h3>
<ul>
    <li><strong>Fine-Tuned on Python Code Datasets</strong> - Trained on a diverse set of Python scripts to enhance code understanding and generation capabilities.</li>
    <li><strong>Enhanced Code Debugging and Explanation</strong> - Capable of identifying errors in code snippets and providing detailed explanations.</li>
    <li> <strong>Optimized for Efficient Inference</strong> - Fine-tuning process focused on reducing latency during code generation tasks.</li>
    <li><strong>Context-Aware Responses</strong> - Generates responses that consider the broader context of the input, leading to more coherent and relevant outputs.</li>
    <li><strong>Seamless Integration</strong> - Easily integrates with development environments to assist developers in real-time coding scenarios.</li>
</ul>

<h3>Model Details:</h3>
<ul>
    <li><strong>Base Model:</strong> Llama 3.2</li>
    <li><strong>Fine-Tuning Method:</strong> Full Model Fine-Tuning</li>
    <li><strong>Deployment:</strong> Hugging Face Spaces</li>
    <li><strong>Use Cases:</strong> Code Generation, Debugging, Code Explanation</li>
</ul>

---

<h2>Repository Contents</h2>
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
        <td><code>model-00001-of-00004.safetensors</code></td>
        <td>Segment 1 of the fine-tuned model weights</td>
    </tr>
    <tr>
        <td><code>model-00002-of-00004.safetensors</code></td>
        <td>Segment 2 of the fine-tuned model weights</td>
    </tr>
    <tr>
        <td><code>model-00003-of-00004.safetensors</code></td>
        <td>Segment 3 of the fine-tuned model weights</td>
    </tr>
    <tr>
        <td><code>model-00004-of-00004.safetensors</code></td>
        <td>Segment 4 of the fine-tuned model weights</td>
    </tr>
    <tr>
        <td><code>special_tokens_map.json</code></td>
        <td>Mapping of special tokens used during fine-tuning</td>
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
pip install transformers torch gradio huggingface_hub safetensors
</pre>

<h3>2. Load the Fine-Tuned Model</h3>
<p>Use the following script to load the <strong>Llama 3.2</strong> fine-tuned model:</p>

<pre>
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_repo = "soureesh1211/finetuned-llama3"
model = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=torch.bfloat16, device_map="auto")
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

<h2> Hugging Face Model & Space</h2>
<ul>
    <li><strong>Fine-Tuned Model:</strong> <a href="https://huggingface.co/soureesh1211/finetuned-llama3">soureesh1211/finetuned-llama3</a></li>
    <li><strong>Live Chatbot:</strong> <a href="https://huggingface.co/spaces/soureesh1211/finetuned-llama3-chatbot">Hugging Face Space</a></li>
</ul>

---

<h2>📜 License</h2>
<p>This project is released under the <strong>Apache-2.0 License</strong>. Feel free to use and modify it.</p>

---

<h2>📢 Acknowledgements</h2>
<ul>
    <li>Llama 3.2: The base model used for fine-tuning.</
::contentReference[oaicite:0]{index=0}
 

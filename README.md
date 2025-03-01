# 💬 Fine-Tuned Llama 3 Model

This project fine-tunes **Meta's Llama 3** model to improve its conversational AI capabilities. The fine-tuned model is deployed using **Gradio** for an interactive chatbot interface.

## 🚀 Model Details
- **Base Model:** `Meta Llama 3`
- **Fine-Tuned Model:** [soureesh1211/finetuned-llama3](https://huggingface.co/soureesh1211/finetuned-llama3)
- **Fine-Tuning Technique:** Standard fine-tuning with parameter optimization
- **Frameworks Used:** `Hugging Face Transformers`, `PEFT`, `Gradio`, `Accelerate`

## 📌 Features
- Fine-tuned for enhanced conversational AI performance.
- Efficient inference with optimized token generation.
- Interactive chatbot deployment using **Gradio**.
- Supports **FP16/BF16** computation for better performance.

## 🛠️ Installation
```bash
pip install -r requirements.txt
```

## 📜 Usage
Run the chatbot using:
```bash
python app.py
```

## 📂 Project Files
- `app.py` - Gradio-based chatbot implementation.
- `llama3_finetuning.ipynb` - Fine-tuning script for the model.
- `requirements.txt` - Dependencies required for running the project.

## 🌐 Model Deployment
The model is deployed on Hugging Face and can be accessed here: [Llama 3 Fine-Tuned Model](https://huggingface.co/soureesh1211/finetuned-llama3)

## 📜 License
This project is released under the **Apache-2.0** license.

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model from Hugging Face Model Hub
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & tokenizer
MODEL_REPO = "soureesh1211/finetuned-llama3"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, torch_dtype=torch.float16 if device == "cuda" else torch.bfloat16, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)

# Define inference function
def generate_response(prompt):
    formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful AI assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract only the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    # Remove any unintended formatting
    response = response.replace("\n", " ").strip()
    return response

# Set up Gradio UI
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt..."),
    outputs=gr.Textbox(),
    title="Fine-Tuned Llama 3.2 Chatbot",
    description="Enter a prompt and get a response from the fine-tuned Llama 3.2 model!"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

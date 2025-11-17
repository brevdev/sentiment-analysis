# Load and run a simple model on GPU

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 Simple GPU Demo: Text Generation")
print("=" * 60)
print("\n🎯 This demo optimized for: 1x NVIDIA L40s or A100 40GB")
print("💡 Works on: Any NVIDIA GPU with 8GB+ VRAM")

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Using device: {device}")

# Load model (suppressing verbose output)
print("\n📥 Loading DistilGPT2...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU explicitly
print(f"🔄 Moving model to {device}...")
model = model.to(device)
_ = model.eval()

# Verify placement
model_device = next(model.parameters()).device
print(f"✅ Model is on: {model_device}")

# Show GPU memory
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"💾 GPU Memory Used: {memory_used:.2f} GB")

# Generate text
print("\n🎨 Generating text...")
prompt = "Brev launchables make AI development"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.time()
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=40,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
generation_time = time.time() - start

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n📝 Result: {generated_text}")
print(f"⚡ Time: {generation_time:.2f}s")

print("\n" + "=" * 60)
print("✅ Demo complete! Your GPU is working.")
print("=" * 60)


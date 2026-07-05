"""
Google Colab Quick-Start Script for LoRA Fine-Tuning.

Copy-paste this entire file into a Colab cell and run it.
Works on Colab free tier (TinyLlama) or T4 (Mistral-7B).

Step 1: Run this script in Colab to fine-tune
Step 2: Download the lora_findoc/ folder
Step 3: Place it in your project root
Step 4: Set USE_FINETUNED_MODEL=true in .env
Step 5: Restart Streamlit
"""

# ============================================================================
# Cell 1: Install dependencies
# ============================================================================
# !pip install -q transformers peft datasets accelerate bitsandbytes trl torch

# ============================================================================
# Cell 2: Run fine-tuning
# ============================================================================

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Switch to Mistral-7B if you have T4/A100
OUTPUT_DIR = "./lora_findoc"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
USE_4BIT = True

# --- Training Data (Financial QA pairs) ---
TRAINING_DATA = [
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: Tesla reported total revenues of $96.77 billion for fiscal year 2023, representing a 19% increase from $81.46 billion in 2022. Automotive revenues were $82.42 billion.\nQuestion: What was Tesla's total revenue in 2023?\n\n### Response:\nTesla reported total revenues of $96.77 billion for fiscal year 2023, a 19% year-over-year increase from $81.46 billion in 2022. Automotive revenues specifically accounted for $82.42 billion of the total."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: Apple Inc. reported net sales of $383.3 billion for fiscal year 2023. iPhone revenue was $200.6 billion, Services revenue reached $85.2 billion. Gross margin was 44.1%.\nQuestion: What were Apple's key revenue figures in 2023?\n\n### Response:\nApple reported net sales of $383.3 billion in FY2023. iPhone remained the largest segment at $200.6 billion, while Services revenue reached $85.2 billion. The company maintained a gross margin of 44.1%."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, analyze the risk factors.\n\n### Input:\nContext: The company faces significant supply chain risks including semiconductor shortages, single-source suppliers for critical components, and geopolitical tensions affecting manufacturing in Asia.\nQuestion: What are the key supply chain risks mentioned?\n\n### Response:\nThe filing identifies several critical supply chain risks: (1) semiconductor shortages impacting production capacity, (2) dependency on single-source suppliers for critical components, (3) geopolitical tensions affecting Asian manufacturing operations."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: Ford Motor Company reported a net loss of $1.98 billion on its EV segment (Model e) in 2023, while the overall company posted net income of $4.35 billion.\nQuestion: How did Ford's EV business perform in 2023?\n\n### Response:\nFord's EV segment (Model e) reported a net loss of $1.98 billion in 2023 while the overall company remained profitable with $4.35 billion in net income."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: NVIDIA reported data center revenue of $47.5 billion in FY2024, up 217% year-over-year. Gross margin expanded to 72.7%. Total revenue reached $60.9 billion.\nQuestion: Summarize NVIDIA's financial performance.\n\n### Response:\nNVIDIA achieved exceptional results in FY2024: total revenue of $60.9 billion with data center revenue of $47.5 billion (up 217% YoY), driven by AI chip demand. Gross margin expanded to 72.7%."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: Microsoft's Intelligent Cloud segment revenue was $96.8 billion in FY2024, with Azure growing 29%. AI services contributed an estimated $10+ billion annualized run rate.\nQuestion: How is Microsoft's cloud business performing?\n\n### Response:\nMicrosoft's Intelligent Cloud generated $96.8 billion in FY2024, with Azure growing at 29%. AI services reached an estimated $10+ billion annualized run rate."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: JPMorgan Chase reported net income of $49.6 billion in 2023, with net revenue of $162.4 billion. Return on equity was 17%.\nQuestion: What was JPMorgan's profitability in 2023?\n\n### Response:\nJPMorgan achieved record profitability in 2023 with net income of $49.6 billion on net revenue of $162.4 billion. Return on equity stood at 17%."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, discuss the trends.\n\n### Input:\nContext: Amazon's AWS segment revenue grew from $62.2 billion in 2022 to $90.8 billion in 2023. Operating margin expanded to 40.5%.\nQuestion: What are the trends in Amazon's cloud business?\n\n### Response:\nAWS shows strong growth: revenue grew 46% from $62.2B to $90.8B. Operating margins expanded to 40.5%, indicating improving unit economics and scale advantages."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, analyze the risk factors.\n\n### Input:\nContext: Climate-related risks include physical risks from extreme weather events and transition risks from evolving regulations including carbon pricing.\nQuestion: What climate risks are disclosed?\n\n### Response:\nThe filing discloses (1) physical risks from extreme weather damaging facilities and supply chains, and (2) transition risks from evolving regulations including potential carbon pricing."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: Alphabet reported total revenues of $307.4 billion in 2023. Google Cloud revenue was $33.1 billion, achieving its first full-year operating profit of $1.7 billion.\nQuestion: How is Alphabet's cloud segment performing?\n\n### Response:\nGoogle Cloud reached $33.1 billion in 2023 and achieved its first full-year operating profit of $1.7 billion, signaling sustainable profitability within Alphabet's $307.4 billion total revenue."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, answer the financial question.\n\n### Input:\nContext: The company's R&D expenses increased by 23% to $14.2 billion in 2023, driven by AI investments. Capital expenditure was $32.3 billion for data center expansion.\nQuestion: What are the company's key investment areas?\n\n### Response:\nThe company is heavily investing in AI: R&D rose 23% to $14.2 billion focused on AI/ML. CapEx reached $32.3 billion, primarily for data center expansion and GPU procurement."""},
    {"text": """### Instruction:\nBased on the SEC filing excerpt, analyze cybersecurity risks.\n\n### Input:\nContext: The company faces cybersecurity risks including ransomware, data breaches, and unauthorized access. The company invested $2.1 billion in cybersecurity in 2023.\nQuestion: What cybersecurity risks does the company disclose?\n\n### Response:\nKey cybersecurity risks include (1) ransomware attacks, (2) data breaches exposing customer data, and (3) unauthorized access. The company invested $2.1 billion in cybersecurity measures in 2023."""},
]

# --- Run Training ---
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import Dataset

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.config.use_cache = False

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"\n📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Dataset
dataset = Dataset.from_list(TRAINING_DATA)
print(f"📚 Training examples: {len(dataset)}")

# Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
)

print("🏋️ Training...")
trainer.train()

# Save
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Done! Adapter saved to: {OUTPUT_DIR}")

# ============================================================================
# Cell 3: Test the fine-tuned model
# ============================================================================

from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16
)
finetuned = PeftModel.from_pretrained(base, OUTPUT_DIR)
finetuned.eval()

test_prompt = """### Instruction:
Based on the SEC filing excerpt, answer the financial question.

### Input:
Context: Tesla reported total revenues of $96.77 billion for fiscal year 2023.
Question: What was Tesla's revenue in 2023?

### Response:
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(finetuned.device)
with torch.no_grad():
    out = finetuned.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True)

print("\n🧪 Test Output:")
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# ============================================================================
# Cell 4: Download adapter (run this, then download from Colab file browser)
# ============================================================================
# !zip -r lora_findoc.zip lora_findoc/
# Then: Place in your project root and set USE_FINETUNED_MODEL=true in .env

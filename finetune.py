# -*- coding: utf-8 -*-
"""
Skrip Fine-tuning dan Merging LoRA untuk Mistral-7B

Dijalankan dari terminal:
1. Untuk training: python finetune.py train
2. Untuk merging:  python finetune.py merge
"""

import torch
import os
import argparse  # Untuk memilih mode train/merge
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# --- Konstanta Global ---
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
DATASET_FILE = "dataset.jsonl"
ADAPTER_DIR = "mistral-hr-lora-adapter"   # Output training, Input merging
MERGED_MODEL_DIR = "merged-mistral-fp16" # Output merging


def run_training():
    """
    Menjalankan proses fine-tuning (QLoRA) dan menyimpan adapter.
    """
    print(f"--- Memulai Proses Training ---")
    print(f"Memulai fine-tuning model: {BASE_MODEL_ID}")
    print(f"Menggunakan dataset: {DATASET_FILE}")
    print(f"Adapter akan disimpan di: {ADAPTER_DIR}")

    # --- 2. Konfigurasi Quantization (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # --- 3. Muat Model Base dan Tokenizer ---
    print("Memuat model base dan tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False  # Nonaktifkan cache untuk training

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4. Konfigurasi LoRA (PEFT) ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    print("Menerapkan adapter LoRA ke model...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # --- 5. Muat dan Format Dataset ---
    print("Memuat dan memformat dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    def create_prompt_format(sample):
        """
        Memformat dataset Anda menjadi satu string prompt.
        Ini adalah format yang akan dipelajari oleh model.
        """
        prompt = f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}"""
        # Tambahkan EOS token di akhir setiap contoh
        return {"text": prompt + tokenizer.eos_token}

    dataset = dataset.map(create_prompt_format)
    print("Dataset berhasil diformat.")

    # --- 6. Tentukan Argumen Training ---
    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,  # Gunakan 16-bit precision
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
    )

    # --- 7. Inisialisasi Trainer (SFTTrainer) ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
        dataset_text_field="text", # Tentukan field teks
        max_seq_length=None, # Biarkan SFTTrainer yang mengurus
    )

    # --- 8. Mulai Training ---
    print("--- Memulai Training LoRA ---")
    trainer.train()
    print("--- Training Selesai ---")

    # --- 9. Simpan Adapter LoRA ---
    print(f"Menyimpan adapter LoRA di {ADAPTER_DIR}...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"✅ Adapter LoRA berhasil disimpan.")


def run_merging():
    """
    Menggabungkan adapter LoRA yang sudah dilatih ke base model
    dan menyimpannya sebagai model full-precision (float16).
    """
    print(f"--- Memulai Proses Merging ---")
    print(f"Model Dasar: {BASE_MODEL_ID}")
    print(f"Adapter LoRA: {ADAPTER_DIR}")
    print(f"Output Model: {MERGED_MODEL_DIR}")

    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

    # --- 2. Muat Base Model (DALAM FLOAT16) ---
    print("\nMemuat base model (Mistral 7B) dalam float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Muat dan Gabungkan Adapter LoRA ---
    print(f"Memuat adapter LoRA dari {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Menggabungkan (merging) adapter ke base model...")
    merged_model = model.merge_and_unload()
    print("Merge selesai.")

    # --- 4. Simpan Model yang Sudah Digabung ---
    print(f"Menyimpan model float16 yang sudah digabung ke {MERGED_MODEL_DIR}...")
    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    print(f"\n✅ Model float16 berhasil digabung dan disimpan di: {MERGED_MODEL_DIR}")
    print("Folder ini sekarang siap untuk dikonversi ke GGUF.")


def main():
    """
    Fungsi utama untuk menjalankan skrip.
    """
    # Siapkan parser untuk argumen command-line
    parser = argparse.ArgumentParser(description="Skrip Fine-tuning atau Merging LoRA.")
    parser.add_argument(
        "action",
        choices=["train", "merge"],
        help="Tindakan yang ingin dilakukan: 'train' untuk fine-tuning, 'merge' untuk menggabungkan adapter."
    )

    args = parser.parse_args()

    # Jalankan fungsi berdasarkan pilihan pengguna
    if args.action == "train":
        run_training()
    elif args.action == "merge":
        run_merging()
    else:
        print("Tindakan tidak valid. Pilih 'train' atau 'merge'.")

if __name__ == "__main__":
    main()
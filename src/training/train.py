from transformers import (
                TrainingArguments,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                DataCollatorWithPadding)
import os
import json
import evaluate
from models.base_model import load_model
from datasets import load_dataset


# Loading configuration 
with open("src/config/config_bert_base_multilingual", "r") as file:
    config = json.load(file)
MODEL_NAME = config["model_name"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
OUTPUT_DIR = config["output_dir"]
LOG_DIR = config["log_dir"]

# Load model
model = load_model(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("csv", data_files={
    "train":"data/processed/train.csv",
    "evaluate" : "data/processed/evaluate.csv"
})

def preprocess(examples):
    text = examples["context"] + " " + examples["prompt"] + " " + examples["answer"]
    return tokenizer(text, truncation=True, max_length=256)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.remove_columns(["id", "context", "prompt", "answer"])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch")

# Metric functions
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def calculate_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    acc = accuracy.compute(predictions=preds, references=p.label_ids)
    f1_score = f1.compute(predictions=preds, references=p.label_ids, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1_score["f1"]}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,      # thư mục lưu checkpoint
    evaluation_strategy="epoch",           # evaluate sau mỗi epoch
    save_strategy="epoch",                 # save checkpoint sau mỗi epoch
    save_total_limit=2,                    # (tùy chọn) chỉ giữ 2 checkpoint gần nhất
    load_best_model_at_end=True,           # load model tốt nhất sau khi train
    metric_for_best_model="f1_macro",      # chọn best dựa trên macro-F1
    greater_is_better=True,                # F1 càng cao càng tốt
    num_train_epochs= NUM_EPOCHS,                    # số epoch train
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=LOG_DIR,            # log để theo dõi bằng TensorBoard
    logging_steps=50                       # in log mỗi 50 step
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["evaluate"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=calculate_metrics
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
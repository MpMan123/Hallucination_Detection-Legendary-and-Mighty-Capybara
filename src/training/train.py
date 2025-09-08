from transformers import (
                TrainingArguments,
                AutoModelForSequenceClassification)
import json
import evaluate




# Loading configuration 
with open("src/confic/config_model_unknown", "r") as file:
    config = json.load(file)
MODEL_NAME = config["model_name"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
OUTPUT_DIR = config["output_dir"]
LOG_DIR = config["log_dir"]

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

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
    model,
    args=training_args,
    train_dataset=,
    eval_dataset=,
    data_collator=,
    processing_class=
)

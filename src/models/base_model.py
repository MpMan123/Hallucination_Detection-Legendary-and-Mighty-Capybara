from transformers import AutoModelForSequenceClassification

def load_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    return model

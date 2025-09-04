# Hallucination_Detection-Legendary-and-Mighty-Capybara

# 📂 Folder Structure
```
hallucination-detection/
├── data/                 
│   ├── preprocess.py          # code xử lý dữ liệu thô
│   ├── dataset_statistics.py  # code phân tích dữ liệu
│   └── ... 
│
├── notebooks/            
│   ├── EDA.ipynb              # notebook khám phá dữ liệu (EDA)
│   ├── baseline.ipynb         # thử baseline model nhanh
│   └── experiments/           # notebook cá nhân (N1, N2...)
│
├── src/                  
│   ├── models/               
│   │   ├── base_model.py      # load backbone LLM (RoBERTa, DeBERTa)
│   │   ├── custom_head.py     # thêm classifier head (MLP, logistic)
│   │   ├── ensemble.py        # code ensemble nhiều model
│   │   └── inference.py       # hàm dự đoán cho input mới
│   │
│   ├── training/             
│   │   ├── train.py           # loop train chính (huggingface Trainer/torch)
│   │   ├── evaluate.py        # tính macro-F1, confusion matrix
│   │   └── utils.py           # hàm phụ trợ (save/load model, metrics)
│   │
│   └── config/               
│       ├── config_roberta.json
│       └── config_deberta.json
│
├── results/              
│   ├── logs/                  # log train
│   ├── checkpoints/           # lưu model đã train
│   └── figures/               # confusion matrix, biểu đồ
│
├── docs/                 
│   ├── labels.md              # định nghĩa No/Intrinsic/Extrinsic
│   ├── experiments.md         # mô tả các thí nghiệm
│   └── report.md              # báo cáo nhóm
│
├── requirements.txt      
└── README.md
```

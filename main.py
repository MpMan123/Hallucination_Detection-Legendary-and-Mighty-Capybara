import subprocess, logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

steps = [
    ["python", "eda.py"],
    ["python", "preprocess.py"],
    ["python", "pipeline.py"]
]

for cmd in steps:
    logging.info("👉 Đang chạy: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("❌ Step lỗi: %s", e)
        sys.exit(1)

logging.info("🎉 Hoàn tất pipeline. File cuối cùng: final_input.csv / final_input.xlsx")

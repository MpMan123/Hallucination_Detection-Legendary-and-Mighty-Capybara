import argparse
import logging
import unicodedata
from pyvi import ViTokenizer
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def remove_punctuation_unicode(text: str) -> str:
    """Xóa toàn bộ punctuation & symbol unicode."""
    return "".join(ch for ch in str(text) if not unicodedata.category(ch).startswith(("P","S")))

def tokenize(text: str) -> str:
    """Tách từ bằng ViTokenizer."""
    return " ".join(ViTokenizer.tokenize(str(text)).split())

def remove_stopwords(text: str, stopwords: set) -> str:
    """Loại bỏ stopwords khỏi text."""
    tokens = str(text).split()
    return " ".join([t for t in tokens if t not in stopwords])

def load_stopwords(path: str) -> set:
    """Đọc stopwords từ file .txt (1 từ/1 dòng)."""
    s = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    s.add(w)
    except FileNotFoundError:
        logging.warning("⚠️ Không tìm thấy file stopwords (%s), bỏ qua bước này.", path)
    s.discard("lại")  # giữ hành vi cũ
    return s

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("Đọc dữ liệu từ %s", args.input)
    df = pd.read_csv(args.input, encoding="utf-8").fillna({"response": ""})
    df['response'] = df['response'].astype(str)

    logging.info("Bước 1: lowercase + remove punctuation")
    df['clean_text'] = df['response'].str.lower().progress_apply(remove_punctuation_unicode)

    logging.info("Bước 2: tokenize")
    df['clean_text'] = df['clean_text'].progress_apply(tokenize)

    logging.info("Bước 3: load stopwords")
    stopwords = load_stopwords(args.stopwords)
    if stopwords:
        logging.info("Đọc %d stopwords", len(stopwords))
        df['clean_text'] = df['clean_text'].progress_apply(lambda t: remove_stopwords(t, stopwords))

    logging.info("Lưu kết quả ra %s và %s", args.out_csv, args.out_xlsx)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    df.to_excel(args.out_xlsx, index=False)

    logging.info("Hoàn tất. Xuất %d dòng.", len(df))
    print(df[['response','clean_text']].head())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="vihallu-public-test.csv")
    p.add_argument("--stopwords", default="vietnamese-stopwords.txt")
    p.add_argument("--out-csv", default="clean_data.csv")
    p.add_argument("--out-xlsx", default="clean_data.xlsx")
    args = p.parse_args()
    main(args)

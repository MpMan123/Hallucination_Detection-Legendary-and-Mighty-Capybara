import argparse
import logging
import pandas as pd

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("Đọc dữ liệu từ %s", args.input)
    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input, encoding="utf-8")

    # ưu tiên clean_text nếu có
    ans_col = "clean_text" if "clean_text" in df.columns else "response"
    logging.info("Sử dụng cột '%s' làm câu trả lời", ans_col)

    df["context"] = df.get("context", "").fillna("").astype(str)
    df["prompt"] = df.get("prompt", "").fillna("").astype(str)
    df[ans_col] = df[ans_col].fillna("").astype(str)

    df["representation"] = (
        "[CTX]" + df["context"] +
        "[PRM]" + df["prompt"] +
        "[ANS]" + df[ans_col]
    )

    logging.info("Lưu file kết quả: %s , %s", args.out_csv, args.out_xlsx)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    df.to_excel(args.out_xlsx, index=False)

    logging.info("Hoàn tất. Xuất %d dòng.", len(df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="clean_data.xlsx")
    p.add_argument("--out-csv", default="final_input.csv")
    p.add_argument("--out-xlsx", default="final_input.xlsx")
    args = p.parse_args()
    main(args)

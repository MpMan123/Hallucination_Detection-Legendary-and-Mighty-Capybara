import pandas as pd 

def main():
    df = pd.read_csv("vihallu-public-test.csv", encoding="utf-8")
    print("Kích thước data:", df.shape)
    print("\nCác cột:", df.columns.tolist())
    print("\n5 dòng đầu tiên:")
    print(df.head())

    print("\nThông tin chi tiết:")
    print(df.info())

    print("\nSố lượng giá trị thiếu theo cột:")
    print(df.isna().sum())

    print("\nThống kê mô tả nhanh:")
    print(df.describe(include="all"))

    if "label" in df.columns:
        print("\nPhân phối nhãn thật:")
        print(df['label'].value_counts())
    if "predict_label" in df.columns:
        print("\nPhân phối nhãn dự đoán:")
        print(df['predict_label'].value_counts())

if __name__ == "__main__":
    main()

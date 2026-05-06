import pandas as pd
from pathlib import Path
import glob

def analyze_results(path_pattern="gsm8k_cot_results[*.csv"):
    # read all results csv files
    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = glob.glob(str(subdir / path_pattern))
            print(f"[{subdir.name}] find {len(files)} files")

            # combine the rows
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(f, encoding="latin1")
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, encoding="gbk")
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)

            # data analysis
            print(f"\n=== accuracy for each prompt type for {subdir.name} ===")
            acc_by_type = df.groupby('prompt_type')['correct'].agg([
                'mean', 'count', 'sum'
            ]).round(4)
            acc_by_type['mean'] = acc_by_type['mean'].map('{:.2%}'.format)
            acc_by_type.columns = ['accu', 'total test', 'accu #']
            print(acc_by_type)

    return df

if __name__ == "__main__":
    df_all = analyze_results()

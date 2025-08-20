import pandas as pd
from transformers import HfArgumentParser

from src import utils
from src.config import ExperimentConfig


# 実験前の準備として元となるデータセットを作成するスクリプト
# このスクリプトが実験の始まりに動く想定
# このスクリプトの処理内容は以下の通り
# 0. 実験idが重複していないかチェック。データ上書きなどのミス防止のため、実験id(`config.experiment_id`)が重複していたら止まるようにする。
# 1. ドッキングスコアが計算されている640論文と化合物のデータのみを抽出
# 2. 640論文データを訓練・開発・テストセットに分割
def main(config: ExperimentConfig):
    if config.experiment_dir.exists():
        raise ValueError(
            f"{config.experiment_dir} already exists. Please remove it or change `experiment_id` before running this script."
        )

    # ランダムシードは実験idから生成されるようになっている
    # このスクリプト内のデータ処理はrandom seedの値にのみ影響を受ける(シャッフルの仕方が変化したり)
    # なので、実験idごとにデータは固有になる(データを消してしまっても実験idを同じにすればもう一度同じデータが作れる...はず)
    utils.set_seed(config.random_seed)

    papers_df = pd.read_json(config.seed_papers_path, lines=True)

    papers_df = papers_df[["paper_id", "pubmed_id", "title", "abstract"]]

    # ここでエラーになる時は該当ファイルをutf-8 encodingにして保存し直すとうまくいくかも
    scores_df = pd.read_csv(config.docking_score_path, encoding="utf-8")

    scores_df = scores_df.dropna(subset=["DockingScore"])
    scores_df = scores_df[["ID", "Name", "Synonyms", "SMILES_Original", "DockingScore"]]
    scores_df.columns = ["paper_id", "name", "synonyms", "smiles", "score"]
    scores_df["synonyms"] = (
        scores_df["synonyms"]
        .fillna("")
        .apply(lambda synonyms: [synonym.strip() for synonym in synonyms.split(";") if synonym.strip()])
    )

    # マージによってドッキングスコアが計算されていないデータは除外される
    df = pd.merge(papers_df, scores_df, on="paper_id")

    # データのシャッフル (pandasでたまに使えるテク)
    df = df.sample(frac=1, random_state=config.random_seed)

    # このデータはある程度信用していいデータ(人間が論文と化合物の対応づけを行っているため)
    utils.save_jsonl(df, config.seed_master_path)

    data = df.to_dict("records")
    num_test_data = int(config.test_ratio * len(data))

    train = data[: -2 * num_test_data]
    val = data[-2 * num_test_data : -num_test_data]
    test = data[-num_test_data:]

    assert len(train) + len(val) + len(test) == len(data), "The number of data is not correct."

    # 訓練・開発・テストセット作成
    config.experiment_dir.mkdir(exist_ok=True, parents=True)
    utils.save_jsonl(train, config.train_path)
    utils.save_jsonl(val, config.val_path)
    utils.save_jsonl(test, config.test_path)


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentConfig,))
    [config] = parser.parse_args_into_dataclasses()
    main(config)

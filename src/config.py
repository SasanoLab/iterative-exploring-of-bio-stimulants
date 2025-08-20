import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import utils


@dataclass
class ExperimentConfig:
    # 論文→ドッキングスコア予測モデルのベースとして使うモデルを指定
    # 以前の検証(NL256の論文)ではPubMedBERTが一番性能良かったのでデフォルトで利用
    docking_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    # 論文→化合物名抽出モデルのベースとして使うモデルを指定
    # NL256論文ではT5-3Bを利用していた
    extract_model_name: str = "t5-3b"

    # 実験結果が格納される大元のディレクトリ
    root_dir: Path = "./data/experiments"

    # 実験ごとに用意するidおよびディレクトリ名
    # 例えば5回実験を繰り返してその平均性能を取りたい時はここを変えつつ5回実験するような用途を想定
    # 実験ごとのディレクトリを作成するのに利用する: 例 `./data/experiments/2022-05-11`
    # おすすめ: とりあえずひとつの実験開始日を入れておくとわかりやすそう
    # この部分を変えると実験をリセットできる
    experiment_id: str = "2024-05-29"

    # 笹野研が共有するシミュレーション候補化合物として選定した化合物のSMLIESが記述されているファイル
    output_nlp_path: Path = ...
    # Site16用のファイルパスは以下

    # ドッキングシミュレーションの結果が格納されているファイルへのパス
    # こちらは実験ごとに作成する必要はない(ドッキングシミュレーションの結果は化合物ごとにほぼ固定)
    output_simulation_path: Path = ...

    # 事前にドッキングスコアが計算されている化合物のデータが格納されたファイルへのパス
    docking_score_path: Path = "./data/DockingScores.site18.Targetmol-L2300-640cpds.csv"
    # Site16用のファイルパスは以下
    # docking_score_path: Path = "./data/DockingScores.site16.Targetmol-L2300-640cpds.csv"

    # PubMedから収集してきた関連論文とその検索元になった論文を入れたファイル
    # `docking_score_path`の各論文の関連論文を引っ張ってきており、1行ごとに検索元論文と検索された論文集合(`related_articles`)が紐づいたデータになっている
    # 1レコードにどんなデータが入ってるかは `./data/seed-papers-example.json` ファイルを参照(先頭1行を取り出してformattingしたもの)
    # このデータが候補化合物--論文の対応づけに使われるため、ここおの論文数を増やせば(`related_articles`を増やしたりそもそも元論文数を増やしたり)、データの増強もできる
    seed_papers_path: Path = "./data/seed-papers.jsonl"

    # 事前に収集した640論文の中で、ドッキングスコアが計算されている化合物に対応する論文が格納されたファイルへのパス
    seed_master_path: Path = "./data/seed-master.jsonl"

    # 化合物名とSMLIESへの対応は不変(のはず)なのでここに保存しておいてPubMed APIへの問い合わせ回数を減らす
    # 変な生成結果(化合物名)に対して「対応する化合物なし」を表現できるように文字列とnullが対応づいている場合がある
    name_to_smiles_path: Path = "./data/name-to-smiles.json"

    # 笹野研によるモデル再学習中に投げるSMILESの数
    # 竹内研のシミュレータを暇させないように、こちらの学習中にも向こうに働いていただく
    # ここは気軽に変えても大丈夫だが、多すぎると竹内研のシミュレーターが追いつかないかもしれない
    # 少なすぎるとこちらのモデル再学習中に向こうの計算が終わってしまって時間が勿体無い
    num_backgroud_work_smlies: int = 100

    # ドッキングシミュレーションに利用する化合物の選定手法
    # random: ランダムに選定(100:0)
    # promising: 有望な化合物のみ(0:100)を選定
    # mix: ランダムと有望な化合物を混ぜて(50:50)選定
    recommend_method: str = "mix"

    # 一度に剪定する化合物の量
    num_recommends: int = 100

    # テストデータの割合
    test_ratio: float = 0.2

    # 各実験ごとに作成されるディレクトリを示すプロパティ
    # 実験依存のデータ(モデルや選定してきた化合物、訓練・開発・テストデータ、擬似訓練データなど)は全てこのディレクトリ以下に保存される
    # 逆にドッキングシミュレーションの結果や化合物--SMILESの対応表などは実験非依存なのでself.root_dir以下に保存されている
    # 本プロジェクト共通のディレクトリについてはここでプロパティとして定義しておく(各所で作りなおすことによるミス防止)
    @property
    def experiment_dir(self) -> Path:
        return self.root_dir / self.experiment_id

    # huggingfaceのモデル名にはスラッシュが含まれているが、ディレクトリ区切りと認識されるとディレクトリ構造が変わってめんどい場合があるのでスラッシュを置き換え
    # これでbert-base-uncasedとmicrosoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltextのディレクトリ階層が一致する
    @property
    def docking_model_save_name(self) -> str:
        return self.docking_model_name.replace("/", "__")

    @property
    def extract_model_save_name(self) -> str:
        return self.extract_model_name.replace("/", "__")

    # ドッキングスコア予測モデル関連のデータを保存しておく場所
    # データは選定手法・モデル名ごとに保存する
    @property
    def docking_save_dir(self) -> Path:
        return self.experiment_dir / "docking" / self.recommend_method / self.docking_model_save_name

    # ドッキングスコア予測モデルのweightとtokenizerを保存しておく場所
    # ここに全ハイパラで検証した最良のモデルが保存される
    @property
    def docking_checkpoint_dir(self) -> Path:
        return self.docking_save_dir / "checkpoint"

    # 640論文を訓練・開発・テストセットに分割したファイルへのパス。割合は`test_ratio`で指定。デフォルト3:1:1
    @property
    def train_path(self) -> Path:
        return self.experiment_dir / "train.jsonl"

    @property
    def val_path(self) -> Path:
        return self.experiment_dir / "val.jsonl"

    @property
    def test_path(self) -> Path:
        return self.experiment_dir / "test.jsonl"

    # 論文→化合物名抽出モデルを保存しておく場所
    # データはモデル名ごとに保存する
    @property
    def extract_save_dir(self) -> Path:
        return self.experiment_dir / "extract" / self.extract_model_save_name

    # 化合物名抽出モデルのweightとtokenizerを保存しておく場所
    # ここに全ハイパラで検証した最良のモデルが保存される
    @property
    def extract_checkpoint_dir(self) -> Path:
        return self.extract_save_dir / "checkpoint"

    # `self.seed_papers_path`の中身を元にして生成されるファイル
    # `self.seed_papers_path`の`reraleted_article`中の各関連論文に対し、T5-3Bをはじめとする化合物名抽出モデルを適用して生成された化合物名を対応づける
    # このファイルには一行ごとに論文とその論文に対応する化合物名(化合物名抽出モデルによる予測結果)が格納される想定
    # 予測された化合物名は正しいとは限らないので、PubChem APIに問い合わせて本当に存在する化合物かどうかを確かめる必要があり、このファイルはこのままでは使えない
    # 実際に実験候補化合物の選定に利用するファイルは一つ下のpropertyである`self.candidates_path`のほう
    # 元となる候補化合物--論文集合は選定手法(`recommend_method`)には依存せず、モデル名にだけ依存している
    @property
    def candidate_papers_with_pred_names_path(self) -> Path:
        return self.extract_save_dir / "candidate-papers-with-pred-names.jsonl"

    # `self.candidate_papers_with_pred_names_path`の中身を元にして生成されるファイル
    # PubChem APIを利用して実際に存在すると判定された化合物とその化合物に対応するSMILESが格納されている
    # 元となる候補化合物--論文集合は選定手法(`recommend_method`)には依存せず、モデル名にだけ依存している
    @property
    def candidates_path(self) -> Path:
        return self.extract_save_dir / "candidates.jsonl"

    # 過去に選定した論文・化合物を保存しておくファイル
    # 化合物選定は選定手法・ドッキングスコア予測モデルに依存しているのでドッキングスコア予測モデルと同じ場所にしまっておく
    @property
    def recommends_path(self) -> Path:
        return self.docking_save_dir / "recommends.jsonl"

    # 実験idに対応したランダムシード
    # 実験idを変えた時にランダムシードを変え忘れて同じ実験を繰り返さないように、実験idからseed値を生成するようにしている
    @property
    def random_seed(self) -> int:
        # `ValueError: Seed must be between 0 and 2**32 - 1` の対策として2**32で割った余りを取る
        return int(hashlib.sha256(self.experiment_id.encode("utf-8")).hexdigest(), 16) % (2**32)

    def __post_init__(self):
        # すでにシミュレーションが終了しているSMILESのset
        # 化合物の重複を避けるために用意しておく
        self.simulated_smiles: pd.DataFrame = pd.read_csv(self.output_simulation_path, header=0)
        self.simulated_smiles: dict[str, float] = {
            smiles: score
            for smiles, score in zip(
                self.simulated_smiles["smiles"],
                self.simulated_smiles["docking_score"],
            )
        }

        # 今回論文に対応づいている化合物名はT5によって生成した文字列なので誤りが含まれる
        # PubChemを利用して生成化合物名に対応するSMILES表現があるかどうかをチェックしているが、APIを叩くと時間がかかるのでできれば削減したい
        # このdictは化合物名(誤りを含む)とSMILESの対応を保持しておくためのもの、ここにすでに存在する化合物名についてはPubChemへの問い合わせを省略する
        self.name_to_smiles: dict[str, str] = utils.load_json(self.name_to_smiles_path)

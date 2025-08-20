import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.utils.checkpoint
from torchmetrics.functional import mean_squared_error, spearman_corrcoef
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
)
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments as HFTrainingArguments

from src import utils
from src.config import ExperimentConfig


@dataclass
class TrainingArgs(HFTrainingArguments):
    output_dir: str = None
    num_train_epochs: int = None
    learning_rate: float = None

    per_device_train_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    dataloader_num_workers: int = 4
    lr_scheduler_type: str = "cosine"

    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    max_seq_len: int = 512

    # tensorboardで実験ログを残しておくとどんな感じで学習が進んでいるかわかって便利
    # 以下のようなコマンドで全ハイパラの実験結果をまとめて見れる
    # `tensorboard --logdir data/experiments/2024-05-10/0/mix/microsoft__BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/train-outputs`
    report_to: str = "tensorboard"
    logging_steps: int = 1
    logging_dir: str = None

    # 最良のモデルを選ぶ際に基準となる指標
    # mseでもいいが最終的には「真に有望な化合物をうまく"ランクづけ"できるか」が測りたいので開発セットの化合物の有望度と同じ順位になれば良くなる指標のスピアマンを利用
    metric_for_best_model: str = "spearman"
    greater_is_better: bool = True

    evaluation_strategy: str = "epoch"
    per_device_eval_batch_size: int = 32

    save_strategy: str = "epoch"
    save_total_limit: int = 1

    ddp_find_unused_parameters: bool = False
    load_best_model_at_end: bool = False  # This is importnant for preventing hangup
    remove_unused_columns: bool = False


@dataclass
class HyperParameters:
    lr: float
    epochs: int


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    max_seq_len: int

    def __call__(self, data_list: list[dict[str, Any]]) -> tuple[BatchEncoding, torch.FloatTensor]:
        title = [d["title"] for d in data_list]
        abstract = [d["abstract"] for d in data_list]
        inputs: BatchEncoding = self.tokenizer(
            title,
            abstract,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        inputs["labels"] = torch.Tensor([d["score"] for d in data_list])
        return inputs


def compute_metrics(eval_pred: EvalPrediction):
    pred_scores = torch.Tensor(eval_pred.predictions.reshape(-1))
    gold_scores = torch.Tensor(eval_pred.label_ids.reshape(-1))
    mse = mean_squared_error(pred_scores, gold_scores).item()
    spearman = spearman_corrcoef(pred_scores, gold_scores).item()
    return {
        "mse": mse,
        "spearman": spearman,
    }


# 与えられたハイパラでモデルの学習・開発セットでの評価まで行う
# 全てのハイパラの組み合わせで実験を行い、スピアマンの順位相関係数最良のモデルのみを最終的に残す
# 最良のモデルはその後実験(シミュレーション)化合物の選定に利用される
def run(
    training_args: TrainingArgs,
    config: ExperimentConfig,
    train_dataset: list[dict],
    val_dataset: list[dict],
):
    tokenizer = AutoTokenizer.from_pretrained(config.docking_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.docking_model_name,
        num_labels=1,
        # 回帰系のタスクでは最終層のdropoutはない方が汎化性能が高いらしい
        classifier_dropout=0,
    )

    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_len=training_args.max_seq_len,
    )

    trainer = HFTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer._load_best_model()

    trainer.save_model()
    trainer.save_state()
    trainer.tokenizer.save_pretrained(training_args.output_dir)

    return trainer.state.best_metric


def generate_hparams():
    for epochs in [5, 10, 20]:
        for lr in [2e-5, 5e-5]:
            yield HyperParameters(epochs=epochs, lr=lr)


def search_best_model(
    save_dir: Path,
    training_args: TrainingArgs,
    config: ExperimentConfig,
    train_dataset: list[dict],
    val_dataset: list[dict],
):
    best_checkpoint = None
    best_spearman = 0
    best_hparams = None

    output_dir = save_dir / "train-outputs"

    # 前回までの結果は邪魔なので消す
    shutil.rmtree(str(output_dir), ignore_errors=True)

    # 全ハイパラを総当たり
    for hparams in tqdm(list(generate_hparams()), leave=False, desc="Hyperparameters"):
        run_name = f"LR{hparams.lr}E{hparams.epochs}"
        run_dir = output_dir / run_name

        training_args.run_name = run_name
        training_args.output_dir = str(run_dir)
        training_args.logging_dir = str(run_dir)

        training_args.learning_rate = hparams.lr
        training_args.num_train_epochs = hparams.epochs

        spearman = run(
            training_args=training_args,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        if spearman > best_spearman:
            # 過去最良性能のモデルを残しておく
            best_checkpoint = training_args.output_dir
            best_spearman = spearman
            best_hparams = hparams

    return best_hparams, best_checkpoint, best_spearman


def create_pseudo_df(
    recommends_path: Path,
    output_simulation_path: Path,
) -> list[dict]:
    global_simulation_df: pd.DataFrame = pd.read_csv(output_simulation_path)

    # ドッキングスコアの計算できなかった化合物を除去
    # シミュレーションのタイムアウトなどでたまに発生する
    global_simulation_df = global_simulation_df.dropna(subset=["docking_score"])
    global_simulation_df = global_simulation_df.rename(columns={"docking_score": "score"})

    print(global_simulation_df)

    if not recommends_path.exists():
        recommends_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_jsonl([], recommends_path)

    recommends_df: pd.DataFrame = utils.load_jsonl_df(path=recommends_path)

    # 過去に選定した化合物が存在しない場合がある(実験開始時など)のでその場合はスキップ
    if len(recommends_df) == 0:
        return []

    # 過去に選定した化合物たちと実際にシミュレーションした化合物たちのデータを結合
    pseudo_df = recommends_df.merge(global_simulation_df, on="smiles", how="left")

    pseudo_df = pseudo_df[["pubmed_id", "title", "abstract", "pred_name", "smiles", "score"]]
    pseudo_df = pseudo_df.dropna(subset=["score"])
    return pseudo_df.to_dict(orient="records")


def main(training_args: TrainingArgs, config: ExperimentConfig):
    utils.set_seed(config.random_seed)

    # ドッキングモデル関連のデータの保存先ディレクトリを再帰的に作成(すでにディレクトリがある場合でもエラーが出ないようにするのがexist_ok)
    config.docking_save_dir.mkdir(exist_ok=True, parents=True)

    # 過去の選定化合物と現在までに得られているシミュレーション結果を結合して擬似訓練データ作成
    pseudo_dataset = create_pseudo_df(
        recommends_path=config.recommends_path,
        output_simulation_path=config.output_simulation_path,
    )

    # 元々の訓練データ(640論文を分割したもの)、このデータは実験全体を通して不変
    train_dataset: list[dict] = utils.load_jsonl(config.train_path)

    print(len(train_dataset), len(pseudo_dataset))

    train_dataset += pseudo_dataset
    random.shuffle(train_dataset)

    # 元々の開発データ(640論文を分割したもの)、このデータも実験全体を通して不変
    val_dataset: list[dict] = utils.load_jsonl(config.val_path)

    # 訓練・開発セットを使って最良のモデルを選ぶ
    best_hparams, best_checkpoint, best_spearman = search_best_model(
        config.docking_save_dir,
        training_args,
        config,
        train_dataset,
        val_dataset,
    )

    shutil.copytree(
        best_checkpoint,
        str(config.docking_checkpoint_dir),
        dirs_exist_ok=True,
    )

    log = {
        "datetime": utils.get_current_timestamp(),
        **vars(best_hparams),
        "spearman": best_spearman,
    }

    utils.log(log, config.docking_save_dir / "log.csv")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgs, ExperimentConfig))
    training_args, config = parser.parse_args_into_dataclasses()
    training_args.seed = config.random_seed
    main(training_args, config)

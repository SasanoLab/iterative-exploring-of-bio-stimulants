import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    HfArgumentParser,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
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
    # `tensorboard --logdir data/experiments/2024-05-10/0/docking/mix/microsoft__BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/train-outputs`
    report_to: str = "tensorboard"
    logging_steps: int = 1
    logging_dir: str = None

    # 最良のモデルを選ぶ際に基準となる指標
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

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

    def __call__(self, data_list: list[dict[str, Any]]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        abstract = [d["abstract"] for d in data_list]
        texts = [f"Title: {t}\nAbstract: {a}" for t, a in zip(title, abstract)]

        inputs: BatchEncoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        names = [d["name"] for d in data_list]
        names: BatchEncoding = self.tokenizer(
            names,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        labels = names.input_ids.clone()
        labels[~names.attention_mask.bool()] = -100
        inputs["labels"] = labels
        return inputs


# 与えられたハイパラでモデルの学習・開発セットでの評価まで行う
# 注意: T5-3Bはモデルの保存が重すぎて学習が止まってるように見える場合がある
# かなり待つと動いているのが確認できると思われるので忍耐力高めでヨロ
def run(
    training_args: TrainingArgs,
    config: ExperimentConfig,
    train_dataset: list[dict],
    val_dataset: list[dict],
):
    tokenizer = AutoTokenizer.from_pretrained(
        config.extract_model_name,
        # T5のtokenizerはfast版が若干buggyなので使わないようにしておく
        use_fast=False,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        config.extract_model_name,
        use_cache=False,
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
    )

    trainer.train()
    trainer._load_best_model()

    trainer.save_model()
    trainer.save_state()
    trainer.tokenizer.save_pretrained(training_args.output_dir)

    return trainer.state.best_metric


def generate_hparams():
    for epochs in [10]:
        for lr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]:
            yield HyperParameters(epochs=epochs, lr=lr)


def search_best_model(
    save_dir: Path,
    training_args: TrainingArgs,
    config: ExperimentConfig,
    train_dataset: list[dict],
    val_dataset: list[dict],
):
    best_checkpoint = None
    best_loss = float("inf")
    output_dir = save_dir / "train-outputs"

    # 前回までの結果は邪魔なので消す
    shutil.rmtree(str(output_dir), ignore_errors=True)

    # 全ハイパラを総当たり
    for hparams in tqdm(list(generate_hparams()), leave=False, desc="Hyperparameters"):
        run_name = f"LR{hparams.lr}E{hparams.epochs}"
        run_dir = output_dir / run_name

        # 諸々の結果が出力されるディレクトリ
        training_args.output_dir = str(run_dir)
        # tensorboard関連の設定
        training_args.run_name = run_name
        training_args.logging_dir = str(run_dir)

        training_args.learning_rate = hparams.lr
        training_args.num_train_epochs = hparams.epochs

        loss = run(
            training_args=training_args,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        if loss < best_loss:
            # 過去最良性能のモデルを残しておく
            # ドッキングスコア予測モデルと異なり評価値として開発損失を用いるので"最小"が"最良"
            best_checkpoint = training_args.output_dir
            best_loss = loss

    return best_checkpoint


def main(training_args: TrainingArgs, config: ExperimentConfig):
    utils.set_seed(config.random_seed)

    config.extract_save_dir.mkdir(parents=True, exist_ok=True)

    # 元々の訓練データ(640論文を分割したもの)、このデータは実験全体を通して不変
    train_dataset: list[dict] = utils.load_jsonl(config.train_path)

    # 元々の開発データ(640論文を分割したもの)、このデータも実験全体を通して不変
    val_dataset: list[dict] = utils.load_jsonl(config.val_path)

    # 訓練・開発セットを使って最良のモデルを選ぶ
    best_checkpoint = search_best_model(
        config.extract_save_dir,
        training_args,
        config,
        train_dataset,
        val_dataset,
    )

    shutil.copytree(
        best_checkpoint,
        str(config.extract_checkpoint_dir),
        dirs_exist_ok=True,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgs, ExperimentConfig))
    training_args, config = parser.parse_args_into_dataclasses()
    training_args.seed = config.random_seed
    main(training_args, config)

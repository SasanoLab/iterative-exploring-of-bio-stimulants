from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    HfArgumentParser,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src import utils
from src.config import ExperimentConfig


@dataclass
class Args:
    max_seq_len: int = 512
    batch_size: int = 32


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    max_seq_len: int

    def __call__(self, data_list: list[dict[str, Any]]) -> tuple[BatchEncoding, torch.FloatTensor]:
        title = [d["title"] for d in data_list]
        abstract = [d["abstract"] for d in data_list]
        return self.tokenizer(
            title,
            abstract,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )


@torch.inference_mode()
def predict_docking_scores(
    args: Args,
    config: ExperimentConfig,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(config.docking_checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.docking_checkpoint_dir).eval().to("cuda")

    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    dataloader = DataLoader(
        candidates_df.to_dict(orient="records"),
        collate_fn=data_collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    pred_scores = []
    for inputs in tqdm(dataloader):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out: SequenceClassifierOutput = model.forward(**inputs.to("cuda"))
        pred_score = out.logits.view(-1)
        pred_scores.append(pred_score.float().cpu())

    pred_scores: torch.Tensor = torch.cat(pred_scores, dim=0)
    return pred_scores.tolist()


@torch.inference_mode()
def run(args: Args, config: ExperimentConfig) -> set[str]:
    # 過去に選定した化合物のリスト
    old_recommends_df: list[dict] = utils.load_jsonl(config.recommends_path)

    # 過去に選定した化合物に対応する論文(のPubMed ID)のset
    # これらの論文はすでにドッキングスコアが計算されているため、再計算する必要はない
    simulated_pubmed_ids = set([d["pubmed_id"] for d in old_recommends_df])

    # 候補化合物--論文集合
    candidates_df: pd.DataFrame = utils.load_jsonl_df(config.candidates_path)

    # すでに選定した化合物(simulated_pubmed_idsに含まれる論文)は候補から除外する
    candidates_df: pd.DataFrame = candidates_df[~candidates_df["pubmed_id"].isin(simulated_pubmed_ids)]

    # 全論文についてドッキングスコアを予測
    pred_docking_scores = predict_docking_scores(
        args,
        config,
        candidates_df,
    )

    candidates_df["pred_score"] = pred_docking_scores

    # ドッキングスコアは小さい方が"良い"のでascending=Trueをつけて昇順に並べる
    candidates_df: pd.DataFrame = candidates_df.sort_values(by="pred_score", ascending=True)

    # 新しく選定する化合物のリスト
    new_recommends_df: list[dict] = candidates_df.to_dict(orient="records")[: config.num_recommends]

    # 選定した有望な化合物を選定した日付を追加しておく(デバッグや分析等への利用を想定)
    datetime = utils.get_current_timestamp()
    new_recommends_df: list[dict] = [
        {
            **d,
            "datetime": datetime,
        }
        for d in new_recommends_df
    ]

    # 新しく選定した化合物を過去の選定化合物リストに追加
    recommends_df: pd.DataFrame = pd.DataFrame(old_recommends_df + new_recommends_df)

    utils.save_jsonl(recommends_df, config.recommends_path)

    # 新しく選定された化合物(setにして重複除去済み)
    smiles_to_be_simulated = set(d["smiles"] for d in new_recommends_df)

    return smiles_to_be_simulated


def main(args: Args, config: ExperimentConfig):
    old_simulated_smiles: set[str] = set(config.simulated_smiles.keys())

    # ドッキングスコア予測モデルを動かしてドッキングスコアの良い(小さい)化合物に対応するSMILESのリストを取得
    smiles_to_be_simulated: set[str] = run(args, config)

    # すでにシミュレーション済みのSMILESは除外
    smiles_to_be_simulated: set[str] = smiles_to_be_simulated - old_simulated_smiles

    updated_smiles: list[str] = list(old_simulated_smiles) + list(smiles_to_be_simulated)

    # シミュレーション済みの化合物のリストを更新
    # 新しい化合物が追加されたら竹内研側でシミュレーションしてくれる
    config.output_nlp_path.write_text("\n".join(updated_smiles))


if __name__ == "__main__":
    parser = HfArgumentParser((Args, ExperimentConfig))
    [args, config] = parser.parse_args_into_dataclasses()
    main(args, config)

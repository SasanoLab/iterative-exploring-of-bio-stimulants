from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, HfArgumentParser, PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer

from src import utils
from src.config import ExperimentConfig


@dataclass
class Args:
    max_seq_len: int = 512
    batch_size: int = 32

    num_beams: int = 5
    max_gen_len: int = 16


def normalize(text: str) -> str:
    return text.lower().replace("-", "").replace(",", "").replace(" ", "").strip()


def make_candidate_papers(seed_papers_path: Path) -> pd.DataFrame:
    seed_papers_df = utils.load_jsonl(seed_papers_path)
    data = []
    for row in seed_papers_df:
        for paper in row["related_articles"]:
            data.append(paper)
    return pd.DataFrame(data)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    max_seq_len: int

    def __call__(self, data_list: list[dict[str, Any]]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        abstract = [d["abstract"] for d in data_list]
        texts = [f"Title: {t}\nAbstract: {a}" for t, a in zip(title, abstract)]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )


@torch.inference_mode()
def main(args: Args, config: ExperimentConfig):
    utils.set_seed(config.random_seed)

    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        config.extract_checkpoint_dir,
        use_fast=False,
    )
    model: T5ForConditionalGeneration = (
        T5ForConditionalGeneration.from_pretrained(config.extract_checkpoint_dir).eval().to("cuda")
    )

    candidate_papers_df = make_candidate_papers(config.seed_papers_path)

    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    dataloader = DataLoader(
        candidate_papers_df.to_dict(orient="records"),
        collate_fn=data_collator,
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred_names: list[str] = []

    for inputs in tqdm(dataloader):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # 入力された論文タイトルとアブストラクトを元に化合物名を予測
            gen_ids = model.generate(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
                num_beams=args.num_beams,
                max_length=args.max_gen_len,
            )
        # batch_sizeの数だけ生成された予測化合物名のリスト
        preds: list[str] = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        pred_names.extend(preds)

    candidate_papers_df["pred_name"] = pred_names
    utils.save_jsonl(candidate_papers_df, config.candidate_papers_with_pred_names_path)


if __name__ == "__main__":
    parser = HfArgumentParser((Args, ExperimentConfig))
    [args, config] = parser.parse_args_into_dataclasses()
    main(args, config)

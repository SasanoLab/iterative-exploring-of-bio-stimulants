import time

import pubchempy as pcp
from tqdm import tqdm
from transformers import HfArgumentParser

from src import utils
from src.config import ExperimentConfig


def normalize(text: str) -> str:
    return text.lower().replace("-", "").replace(",", "").replace(" ", "").strip()


def main(config: ExperimentConfig):
    name_to_smiles = config.name_to_smiles
    candidates_names: list[dict] = utils.load_jsonl(config.candidate_papers_with_pred_names_path)
    candidates = []

    for candidate in tqdm(candidates_names, dynamic_ncols=True):
        pred_name = candidate["pred_name"]
        normalized_name = normalize(pred_name)

        if normalized_name is None or normalized_name == "" or pred_name is None or pred_name == "":
            continue

        # すでにname_to_smilesに化合物とSMILESの対応が存在している場合
        if normalized_name in name_to_smiles:
            if name_to_smiles[normalized_name] is None:
                continue
            else:
                candidates.append(
                    {
                        **candidate,
                        "smiles": name_to_smiles[normalized_name],
                    }
                )
                continue

        # name_to_smilesに化合物とSMILESの対応が存在していない場合
        # PubChem APIに問い合わせを行ってSMILESを取得する
        time.sleep(0.5)
        compounds = pcp.get_compounds(pred_name, "name")

        # 対応する化合物名がない場合
        # 主にpred_nameが誤っている場合が多い(そのほかのケースがあるかはわからない)
        if len(compounds) == 0:
            name_to_smiles[normalized_name] = None
            continue

        # 最初の化合物を取得して利用する
        compound: pcp.Compound = compounds[0]

        # SMILESにはいくつか表現形式が存在するが、今回は立体構造を一意に決定可能なisometric SMILESを利用する
        smiles: str = compound.isomeric_smiles.strip()

        # SMILESが存在しない場合もある
        if not smiles:
            name_to_smiles[normalized_name] = None
            continue

        name_to_smiles[normalized_name] = smiles

        candidates.append(
            {
                **candidate,
                "smiles": smiles,
            }
        )

    # 無事 論文--化合物名--SMILESの対応が取れた化合物たちを保存
    utils.save_jsonl(candidates, config.candidates_path)

    # name_to_smilesを上書き保存
    utils.save_json(name_to_smiles, config.name_to_smiles_path)


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentConfig,))
    [config] = parser.parse_args_into_dataclasses()
    main(config)

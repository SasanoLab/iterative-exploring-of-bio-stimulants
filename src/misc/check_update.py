import sys

import pandas as pd
from transformers import HfArgumentParser

from src.config import ExperimentConfig


# 竹内研側のドッキングシミュレーションが終わっているか確認するスクリプト
# シミュレーション結果を格納するファイルが更新されていたら「異常終了」→ exit code が 1
# シミュレーション結果を格納するファイルが更新されていなかったら「正常終了」→ exit code が 0
# このスクリプトのexit codeを見て後続のスクリプトでモデルの再学習を走らせるかどうか決定する
def main(config: ExperimentConfig):
    simulation_df = pd.read_csv(config.output_simulation_path)
    smiles = config.output_nlp_path.read_text().splitlines()

    updated = len(simulation_df) == len(smiles)

    if updated:
        # re-train scoring model
        sys.exit(1)
    else:
        # do nothing
        sys.exit(0)


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentConfig,))
    [config] = parser.parse_args_into_dataclasses()
    main(config)

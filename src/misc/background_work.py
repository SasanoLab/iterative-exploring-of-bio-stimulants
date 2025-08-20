import random

from transformers import HfArgumentParser

from src.config import ExperimentConfig


# 笹野研でモデルの再学習中に竹内研のシミュレーターを遊ばせておくのはもったいないので、いい化合物かどうかはわからないが投げる
# シミュレーションされていないSMILESを探し、ランダムに選択してoutput_nlp_pathに追加する
def main(config: ExperimentConfig):
    smiles = [s for s in config.name_to_smiles.values() if s not in config.simulated_smiles and s is not None]
    if len(smiles) == 0:
        return

    if len(smiles) >= config.num_backgroud_work_smlies:
        smiles = random.sample(smiles, config.num_backgroud_work_smlies)

    new_smlies = list(config.simulated_smiles) + list(smiles)
    config.output_nlp_path.write_text("\n".join(new_smlies))


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentConfig,))
    [config] = parser.parse_args_into_dataclasses()
    main(config)

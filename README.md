# bio-stimulant

論文: [論文テキストを用いた化合物探索の漸進的効率化](https://ipsj.ixsq.nii.ac.jp/records/226105), NL256

## 概要


1. データセットの分割・準備
    - コマンド: `python src/prepare.py`
2. 化合物名抽出モデルの学習・推論
    1. 化合物名抽出モデルの学習 (所要時間: 6~10時間)
        - コマンド: `accelerate launch --config_file accelerate.json src/extract/train.py`
    2. 学習済みの化合物名抽出モデルを用いて論文から化合物名を抽出 (所要時間: 約8時間)
        - コマンド: `python src/extract/predict_compound_names.py`
    3. 化合物名抽出モデルによって抽出された化合物名とPubChem APIを用いて化合物のSMILESを対応づけ
        - コマンド: `python src/extract/align_compounds_and_smiles.py`
3. ドッキングスコア予測モデルの学習
    - コマンド: `accelerate launch --config_file accelerate.json src/docking/train.py`
4. 学習済みドッキングスコア予測モデルを用いて化合物のドッキングスコアを予測、化合物を選定
    - コマンド: `python src/recommend.py`


## Instllation

```bash
rye sync -f
```

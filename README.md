# FaithfulPersona: Balancing Faithfulness and Personalization in Code Explanations through Self-Critique

## Introduction
This is the repo of the paper, **FaithfulPersona: Balancing Faithfulness and Personalization in Code Explanations through Self-Critique**.

This repository provides scripts for generating personalized and faithful code explanations and evaluating their effectiveness using multiple metrics.

You can generate explanations and evaluate them using the scripts in `generation_evaluation`.

User profiles are available in `user_profile`.

## Usage

### Run Explanation Generation
```bash
python all_flow_disco.py --sample_num 1 --iter_count 3 --group valid --mode explain

python all_flow_disco_personalization.py --sample_num 1 --group valid --mode explain --user_id 179736 --iter_count 3

python all_flow_base_personalization.py --sample_num 3 --group valid --mode explain --user_id 179736

python all_flow_base_personalization_consistency.py --sample_num 3 --group valid --mode explain --user_id 179736
```

### Run Pass@K Evaluation
```bash
python all_flow_disco.py --sample_num 1 --iter_count 3 --group valid --mode evaluate

python all_flow_disco_personalization.py --sample_num 1 --group valid --mode evaluate --user_id 179736 --iter_count 3

python all_flow_base_personalization.py --sample_num 3 --group valid --mode evaluate --user_id 179736

python all_flow_base_personalization_consistency.py --sample_num 3 --group valid --mode evaluate --user_id 179736
```

### Run Win Rate Evaluation
```bash
python win_rate.py --group valid --sample_num 1 --user_id 179736 --method base_personalization --iter_count 3
```

### Run Word Overlap Evaluation
```bash
python word_overlap.py --group valid --sample_num 1 --user_id 179736 --method disco_personalization --iter_count 3
```

### Run Rouge-L Evaluation
```bash
python rouge_l.py --group valid --sample_num 1 --user_id 179736 --method disco_personalization --iter_count 3
```

## Citation
If you find this repository useful, please cite our paper:
```bibtex
@article{luofaithfulpersona,
  title={FaithfulPersona: Balancing Faithfulness and Personalization in Code Explanations through Self-Critique},
  author={Luo, Zhuang and Li, Yichuan and Xu, Zexing and Lee, Kyumin and Etesami, S Rasoul}
}

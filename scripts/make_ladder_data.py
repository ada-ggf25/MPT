"""Build data/ladder/{train,val}.parquet for MPT on LADDER's SimplerEnv-Long: one row per (task, layout), in the schema of the other
settings' parquets. The trainer pops env_kwargs and posts it to the env server's /reset, which builds LADDER's episode ``episode``
(0-23, LADDER's fixed layouts) of ``task``.

The split is by TASK (LADDER has none; results declare ``tuned_on_ladder``): train rows are the training tasks of the phase batch
(Vault Exp-2026-11-25-Phase-Batch-Preregistration), val rows are the eight held-out tasks of L5 and L5b, never trained on.

    python scripts/make_ladder_data.py [--out data/ladder] [--layouts 24]
"""
import argparse
import os

TRAIN_TASKS = ("all2_towel split2 relay2e all3_plate all3_towel all4_towel all6_plate move_plate_to_towel_3 move_towel_to_plate_3 "
               "ordered_plate_2 relay3e relay5e split4 split6 spoons3").split()
VAL_TASKS = "all2_plate ordered_plate_3 split3 except_eggplant_4 relay7e split5 all5_plate except_eggplant_5".split()


def row(split, index, task, episode):
    return {"data_source": "ladder", "ability": "ladder",
            "reward_model": {"style": "rule", "ground_truth": "subgoals"},
            "prompt": [{"role": "user", "content": task}],
            "extra_info": {"split": split, "index": index, "task": task, "episode": episode},
            "env_kwargs": {"task": task, "episode": episode, "seed": episode, "data_source": "ladder"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "ladder"))
    ap.add_argument("--layouts", type=int, default=24)
    args = ap.parse_args()
    import pandas as pd

    os.makedirs(args.out, exist_ok=True)
    for split, tasks in (("train", TRAIN_TASKS), ("val", VAL_TASKS)):
        rows = [row(split, len(tasks) * 0 + i, t, k) for i, (t, k) in
                enumerate((t, k) for t in tasks for k in range(args.layouts))]
        pd.DataFrame(rows).to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(rows)} rows -> {args.out}/{split}.parquet")


if __name__ == "__main__":
    main()

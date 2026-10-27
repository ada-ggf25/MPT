"""Build data/libero/{train,val}.parquet: one row per (task, initial state), in the schema of the
other settings' parquets (data_source / ability / reward_model / prompt / extra_info / env_kwargs).
The trainer pops env_kwargs and posts it to the env server's /reset.

    python scripts/make_libero_data.py --suite libero_10 --train-inits 0-39 --val-inits 40-49
    python scripts/make_libero_data.py --fake      # rows for eom-robotics' FakeBackend (no LIBERO)

Without --fake, the task texts come from LIBERO itself (pip install libero in this env).
"""
import argparse
import os


def parse_range(s):
    lo, hi = (int(x) for x in s.split("-"))
    return list(range(lo, hi + 1))


def libero_tasks(suite):
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[suite]()
    return [(i, bench.get_task(i).language) for i in range(bench.n_tasks)]


def fake_tasks():
    # mirrors eom_robotics.stage2.backends.FAKE_TASKS (task name -> text)
    return [("bowl_drawer", "open the top drawer, then put the black bowl in it"),
            ("mug_plate", "put the white mug on the plate"),
            ("stove_pot", "turn on the stove, then put the moka pot on it")]


def row(source, split, index, text, env_kwargs):
    return {"data_source": source, "ability": "libero",
            "reward_model": {"style": "rule", "ground_truth": "goal_conditions"},
            "prompt": [{"role": "user", "content": text}],
            "extra_info": {"split": split, "index": index, "task": source, "instruction": text},
            "env_kwargs": env_kwargs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--train-inits", default="0-39")
    ap.add_argument("--val-inits", default="40-49")
    ap.add_argument("--max-steps-per-turn", type=int, default=150)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "libero"))
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(args.out, exist_ok=True)
    for split, inits in (("train", parse_range(args.train_inits)), ("val", parse_range(args.val_inits))):
        rows = []
        if args.fake:
            for k, (name, text) in enumerate(fake_tasks()):
                for j in inits:
                    rows.append(row("libero_fake", split, len(rows), text,
                                    {"task": name, "seed": 1000 * k + j, "data_source": "libero_fake"}))
        else:
            for task_id, text in libero_tasks(args.suite):
                for j in inits:
                    rows.append(row(args.suite, split, len(rows), text,
                                    {"suite": args.suite, "task_id": task_id, "init_id": j, "seed": j,
                                     "max_steps_per_turn": args.max_steps_per_turn,
                                     "data_source": args.suite}))
        pd.DataFrame(rows).to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(rows)} rows -> {args.out}/{split}.parquet")


if __name__ == "__main__":
    main()

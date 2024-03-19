import os
import pandas as pd
import json
import argparse


def dirs(dir: str):
    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            yield file


def files(dir: str):
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            yield file


def create_table(func_dirs: list[str], parent_dir: str):
    fields = [
            "func",
            "N",
            "threads",
            "bs",
            "eps",
            "t",
            "iters",
            "max_err",
            "aver_err",
            "std_err",
            "max_init_val",
            "mes",
            "img",
        ]
    df = pd.DataFrame(
        columns=fields
    )
    for dir in func_dirs:
        cur_dir = os.path.join(parent_dir, dir)
        for test in dirs(cur_dir):
            img_path = None
            cfg = None
            test_dir = os.path.join(cur_dir, test)
            for fl in files(test_dir):
                cur_fl = os.path.join(test_dir, fl)
                if fl[-5:] == ".json":
                    with open(cur_fl) as json_data:
                        cfg = json.load(json_data)
                if fl[-4:] == ".png":
                    img_path = os.path.join(dir, test, fl)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "func": dir,
                                "N": cfg["N"],
                                "eps": cfg["eps"],
                                "threads": cfg["thr_n"],
                                "max_init_val": cfg["max_init"],
                                "bs": cfg["bs"],
                                "t": cfg["time"],
                                "iters": cfg["iters"],
                                "max_err": cfg["max_err"],
                                "aver_err": cfg["aver_err"],
                                "std_err": cfg["std_err"],
                                "mes": cfg["spec_info"],
                                "img": img_path,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return df.sort_values(fields, ascending=[True for _ in fields])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir")
    parser.add_argument("save_dir")
    args = parser.parse_args()

    test_dirs = [d for d in dirs(args.parent_dir)]
    res = create_table(test_dirs, args.save_dir)

    res.to_csv(os.path.join(args.save_dir, "REPORT.csv"), index=False)
    res.to_html(os.path.join(args.save_dir, "REPORT.html"), index=False)
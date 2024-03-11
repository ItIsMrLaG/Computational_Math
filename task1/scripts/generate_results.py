import os 
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse

def generate_preview(res_m: np.ndarray, ans_m: np.ndarray, config: dict):
    h, v = 50, 120
    x = np.linspace(0, 1, config["N"])
    y = np.linspace(0, 1, config["N"])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(dpi=150.0, figsize=plt.figaspect(0.5))
    fig.suptitle(config["spec_info"], ha='center', fontsize=16)
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Approximation")
    ax.plot_surface(X, Y, res_m, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(h, v)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Function")
    ax.plot_surface(X, Y, ans_m, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(h, v)
    return fig

def get_path(dir: str):
    def files():
        for file in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, file)):
                yield file

    res, ans, met = {}, {}, {}
    inds = set()

    for name in files():
        if name.find("res") != -1:
            i = name.replace("res_", "")
            i = int(i.replace(".csv", ""))
            inds.add(i) 
            res[i] = str(os.path.join(dir, name))

        elif name.find("ans") != -1:
            i = name.replace("ans_", "")
            i = int(i.replace(".csv", ""))
            inds.add(i)
            ans[i] = str(os.path.join(dir, name))

        elif name.find("met") != -1:
            i = name.replace("met_", "")
            i = int(i.replace(".json", ""))
            inds.add(i)
            met[i] = str(os.path.join(dir, name))
    return list(inds), res, ans, met

def generate_result(res_p: str, ans_p: str, sav_p: str, img: bool, cfg: dict):
    ans_matrix = pd.read_csv(ans_p, sep=',', header=None).values
    res_matrix = pd.read_csv(res_p, sep=',', header=None).values

    d = np.abs(res_matrix - ans_matrix)
    max_err = np.max(d)
    aver_err = np.average(d)
    std_err = np.std(d)
    cfg["max_err"] = round(max_err, 2)
    cfg["aver_err"] = round(aver_err, 2)
    cfg["std_err"] = round(std_err, 2)

    ex_name = f"gs={cfg["N"]}_th={cfg["thr_n"]}_bs={cfg["bs"]}_ep={cfg["eps"]}_l={cfg["side_l"]}_rn={cfg["max_init"]}_ms={cfg["spec_info"]}_{round(time.time())}"
    ex_path = os.path.join(sav_p, ex_name)
    if not os.path.exists(ex_path):
        os.mkdir(ex_path)

    if img:
        img = generate_preview(res_matrix, ans_matrix, cfg)
        img.savefig(str(os.path.join(ex_path, "img.png")))

    with open(os.path.join(ex_path, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir")
    parser.add_argument("-i", "--img", type=int, help="Write name of dir (function)")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    keys, r, a, m = get_path("experiments")
    
    for key in keys:
        with open(m[key]) as json_data:
            cfg = json.load(json_data)

        main_path = os.path.join(args.save_dir, cfg["spec_info"])
        if not os.path.exists(main_path):
            os.mkdir(main_path)

        generate_result(r[key], a[key], main_path, bool(args.img), cfg)

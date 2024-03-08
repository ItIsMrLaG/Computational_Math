import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


def generate_preview(res_m: np.ndarray, ans_m: np.ndarray, config: dict, func_name:str =None):
    x = np.linspace(0, 1, config["N"])
    y = np.linspace(0, 1, config["N"])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(dpi=150.0, figsize=plt.figaspect(0.5))
    fig.suptitle(func_name, ha='center', fontsize=16)
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Approximation")
    ax.plot_surface(X, Y, res_m, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.view_init(50, 100)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Function")
    ax.plot_surface(X, Y, ans_m, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.view_init(50, 100)

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

def generate_result(res_p: str, ans_p: str, met_p: str, sav_p: str):
    with open(met_p) as json_data:
        cfg = json.load(json_data)

    ans_matrix = pd.read_csv(ans_p, sep=',', header=None).values
    res_matrix = pd.read_csv(res_p, sep=',', header=None).values
    max_err = np.max(np.abs(res_matrix - ans_matrix))
    aver_err = np.average(np.abs(res_matrix - ans_matrix))
    cfg["max_err"] = round(max_err, 2)
    cfg["aver_err"] = round(aver_err, 2)

    ex_name = f"gs={cfg["N"]}_th={cfg["thr_n"]}_bs={cfg["bs"]}_eps={cfg["eps"]}"
    ex_path = os.path.join(sav_p, ex_name)
    if not os.path.exists(ex_path):
        os.mkdir(ex_path)

    img = generate_preview(ans_matrix, res_matrix, cfg, func_name)
    img.savefig(str(os.path.join(ex_path, "img.png")))
    with open(os.path.join(ex_path, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f)

if __name__ == '__main__':
    func_name = "1000*(x^3)+2000*(y^3)"
    main_path = os.path.join("./", func_name)

    keys, r, a, m = get_path("experiments")
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    for key in keys:
        generate_result(r[key], a[key], m[key], main_path)


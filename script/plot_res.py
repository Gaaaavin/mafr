import seaborn as sns
import matplotlib.pyplot as plt
import os
import json


checkpoint_dirs = [
    "../res/arcface_wf",
    "../res/arcface_lfw",
    "../res/arcface_dist_wf",
    "../res/arcface_dist_lfw"
]
for checkpoint_dir in checkpoint_dirs:
    res_path = os.path.join(checkpoint_dir, "results.json")
    with open(res_path, 'rt') as f:
        results = json.load(f)

    # results = pandas.DataFrame.from_dict(results)
    del results["loss"]
    sns.lineplot(results)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(checkpoint_dir.split("/")[-1])
    plt.savefig(os.path.join(checkpoint_dir, "results.pdf"))
    plt.close()

import seaborn as sns
import matplotlib.pyplot as plt
import os
import json


# checkpoint_dirs = [
#     "../res/arcface_30k_p0",
#     "../res/arcface_30k_p0_dist",
#     "../res/arcface_30k_p5",
#     "../res/arcface_30k_p5_dist"
# ]
checkpoint_dirs = os.listdir("../res")
for checkpoint_dir in checkpoint_dirs:
    res_path = os.path.join("../res", checkpoint_dir, "results.json")
    with open(res_path, 'rt') as f:
        results = json.load(f)

    # results = pandas.DataFrame.from_dict(results)
    del results["loss"]
    sns.lineplot(results)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(checkpoint_dir.split("/")[-1])
    plt.savefig(os.path.join("fig", checkpoint_dir+".pdf"))
    plt.close()

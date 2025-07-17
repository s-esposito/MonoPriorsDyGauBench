import pickle
from collections import defaultdict


def create_markdown_table(data, title):
    table = (
        f"| Method | PSNR↑ | SSIM↑ | LPIPS↓ |\n|--------|-------|-------|--------|\n"
    )
    for row in data:
        table += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n"
    table += f"\n{title}\n\n"
    return table


datasets = ["nerfies_hypernerf", "iphone"]
methods = [
    "TiNeuVox/vanilla",
    "MLP/nodeform",
    "MLP/vanilla",
    "Curve/vanilla",
    "FourDim/vanilla",
    "HexPlane/vanilla",
    "TRBF/nodecoder",
    "TRBF/vanilla",
]

methods_to_show = [
    "TiNeuVox",
    "3DGS",
    "DeformableGS",
    "EffGS",
    "RTGS",
    "4DGS",
    "STG-decoder",
    "STG",
]

with open("traineval.pkl", "rb") as file:
    result_final = pickle.load(file)

tables = {}

for dataset in datasets:
    if dataset not in tables:
        tables[dataset] = []
    for method, method_show in zip(methods, methods_to_show):
        scene_metrics = defaultdict(list)

        if dataset == "nerfies_hypernerf":
            datasets_to_combine = ["nerfies", "hypernerf"]
        else:
            datasets_to_combine = [dataset]

        for sub_dataset in datasets_to_combine:
            if method not in result_final[sub_dataset]:
                continue

            for scene in result_final[sub_dataset][method]:
                if (
                    scene == "all"
                    or len(result_final[sub_dataset][method][scene]["test_psnr"]) == 0
                ):
                    continue

                psnr = sum(
                    [
                        x[0]
                        for x in result_final[sub_dataset][method][scene]["test_psnr"]
                    ]
                ) / len(result_final[sub_dataset][method][scene]["test_psnr"])
                ssim = sum(
                    [
                        x[0]
                        for x in result_final[sub_dataset][method][scene]["test_ssim"]
                    ]
                ) / len(result_final[sub_dataset][method][scene]["test_ssim"])
                lpips = sum(
                    [
                        x[0]
                        for x in result_final[sub_dataset][method][scene]["test_lpips"]
                    ]
                ) / len(result_final[sub_dataset][method][scene]["test_lpips"])

                scene_metrics["psnr"].append(psnr)
                scene_metrics["ssim"].append(ssim)
                scene_metrics["lpips"].append(lpips)

        if scene_metrics["psnr"]:  # Check if we have any data for this method
            avg_psnr = sum(scene_metrics["psnr"]) / len(scene_metrics["psnr"])
            avg_ssim = sum(scene_metrics["ssim"]) / len(scene_metrics["ssim"])
            avg_lpips = sum(scene_metrics["lpips"]) / len(scene_metrics["lpips"])

            tables[dataset].append(
                (
                    method_show,
                    round(avg_psnr, 2),
                    round(avg_ssim, 3),
                    round(avg_lpips, 3),
                )
            )

# Create markdown content
markdown_content = """
| Method | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|--------|--------|--------|---------|
| T-NeRF | 21.55 | 0.595 | 0.297 |
| NSFF [4] | 19.53 | 0.521 | 0.471 |
| Nerfies [*] | 20.85 | 0.562 | 0.200 |
| HyperNeRF [7] | 21.16 | 0.565 | **0.192** |

Table 2: Benchmark results on the rectified Nerfies-HyperNeRF dataset. Please see the Appendix for the breakdown over 7 multi-camera sequences.

| Method | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|--------|--------|--------|---------|
| T-NeRF | 16.96 | 0.577 | 0.379 |
| NSFF [4] | 15.46 | 0.551 | 0.396 |
| Nerfies [*] | 16.45 | 0.570 | 0.339 |
| HyperNeRF [7] | 16.81 | 0.569 | 0.332 |

Table 3: Benchmark results on the proposed iPhone dataset. Please see the Appendix for the breakdown over 7 multi-camera sequences of complex motion.

"""

# Add new tables
markdown_content += create_markdown_table(
    tables["nerfies_hypernerf"],
    "Table 4: Benchmark results on the combined Nerfies-HyperNeRF dataset.",
)
markdown_content += create_markdown_table(
    tables["iphone"], "Table 5: Benchmark results on the proposed iPhone dataset."
)

# assert False, markdown_content
# Save markdown content to file
with open("benchmark_results.md", "w") as f:
    f.write(markdown_content)

print("Markdown file 'benchmark_results.md' has been created successfully.")

# Save markdown content to file
with open("dycheck_drop.md", "w") as f:
    f.write(markdown_content)

print("Markdown file 'benchmark_results.md' has been created successfully.")

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

ds = load_dataset("adlito/covid19-cxr-bm3d-denoising")
output_dir = Path("/workspace/images")

image_cols = ["original_image", "noisy_image", "bm3d_image"]

# Create all subdirectories first
for col in image_cols:
    (output_dir / col.split("_")[0]).mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(tqdm(ds["train"])):
    img_id = sample["image_id"]
    label  = sample["label"]

    for col in image_cols:
        out_path = output_dir / col.split("_")[0] / f"{label}_{img_id}_{i}.png"
        sample[col].save(out_path)
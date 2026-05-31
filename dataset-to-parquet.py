from pathlib import Path

from datasets import Dataset, Features, Image, Value


if __name__ == "__main__":
    dataset_path = Path("data")

    data = []

    for noisy_image_path in (dataset_path / "noisy").glob("*.png"):
        label, image_id, sigma = noisy_image_path.stem.split("-")
        image_id = int(image_id)
        sigma = int(sigma)
        original_image_name = f"{label}-{image_id}"

        data.append({
            "original_image": f"data/original/{original_image_name}.png",
            "noisy_image": f"data/noisy/{noisy_image_path.stem}.png",
            "bm3d_image": f"data/bm3d/{noisy_image_path.stem}.png",
            "label": label,
            "image_id": image_id,
            "noise_level": (sigma / 10000.) * 255
        })
    
    features = Features({
        "original_image": Image(),
            "noisy_image": Image(),
            "bm3d_image": Image(),
            "label": Value("string"),
            "image_id": Value("int32"),
            "noise_level": Value("float32")
    })

    ds = Dataset.from_list(data, features=features)
    ds.push_to_hub("adlito/covid19-cxr-bm3d-denoising")

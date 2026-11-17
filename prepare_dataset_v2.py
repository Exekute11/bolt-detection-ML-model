from pathlib import Path
import random
import shutil


# ============================================================
# PATHS
# ============================================================

CURRENT_PROJECT = Path(__file__).resolve().parent

ORIGINAL_DATASET = Path(
    r"C:\Users\dniphade\Documents\BoltDetectionAI_Final"
    r"\HexBoltDataset_Raw\HexBoltDataset_YOLO"
)

NEW_DATA_ROOT = CURRENT_PROJECT / "HexBoltDataset_V2_Raw"

FRONT_IMAGES = NEW_DATA_ROOT / "Front_Missing_2"
FRONT_LABELS = NEW_DATA_ROOT / "Front_Missing_2_Labels"

END_IMAGES = NEW_DATA_ROOT / "End_Missing_1"
END_LABELS = NEW_DATA_ROOT / "End_Missing_1_Labels"

OUTPUT_DATASET = CURRENT_PROJECT / "HexBoltDataset_V2_YOLO"

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def create_output_structure():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in ("train", "val"):
        (OUTPUT_DATASET / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DATASET / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def image_files(folder):
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def validate_label_file(label_path):
    """
    Validate basic YOLO detection-label format:

    class_id x_center y_center width height
    """

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        raise ValueError(
            f"Empty annotation file: {label_path}"
        )

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO row in {label_path.name}, "
                f"line {line_number}: {line}"
            )

        class_id = int(float(parts[0]))

        if class_id != 0:
            raise ValueError(
                f"Unexpected class ID {class_id} in "
                f"{label_path.name}. Expected class 0."
            )

        coordinates = [
            float(value)
            for value in parts[1:]
        ]

        if not all(
            0.0 <= value <= 1.0
            for value in coordinates
        ):
            raise ValueError(
                f"Coordinate outside 0 to 1 in "
                f"{label_path.name}, line {line_number}"
            )

    return len(lines)


def copy_pair(
    image_path,
    label_path,
    split,
    destination_stem,
):
    destination_image = (
        OUTPUT_DATASET
        / "images"
        / split
        / f"{destination_stem}{image_path.suffix.lower()}"
    )

    destination_label = (
        OUTPUT_DATASET
        / "labels"
        / split
        / f"{destination_stem}.txt"
    )

    shutil.copy2(image_path, destination_image)
    shutil.copy2(label_path, destination_label)


def copy_original_dataset():
    copied = {
        "train": 0,
        "val": 0,
    }

    for split in ("train", "val"):
        source_images = (
            ORIGINAL_DATASET
            / "images"
            / split
        )

        source_labels = (
            ORIGINAL_DATASET
            / "labels"
            / split
        )

        for image_path in image_files(source_images):
            label_path = (
                source_labels
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Original label missing: {label_path}"
                )

            validate_label_file(label_path)

            copy_pair(
                image_path=image_path,
                label_path=label_path,
                split=split,
                destination_stem=(
                    f"original_{split}_"
                    f"{image_path.stem}"
                ),
            )

            copied[split] += 1

    return copied


def load_new_samples(
    image_folder,
    label_folder,
    prefix,
):
    samples = []

    for image_path in image_files(image_folder):
        label_path = (
            label_folder
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"New label missing: {label_path}"
            )

        box_count = validate_label_file(
            label_path
        )

        samples.append(
            {
                "image": image_path,
                "label": label_path,
                "prefix": prefix,
                "box_count": box_count,
            }
        )

    return samples


def split_new_samples(samples):
    """
    Put approximately 80% of each missing-bolt condition
    into train and 20% into validation.
    """

    shuffled = list(samples)

    random.Random(
        RANDOM_SEED
    ).shuffle(shuffled)

    validation_count = max(
        1,
        round(len(shuffled) * 0.20),
    )

    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]

    return training, validation


def copy_new_samples(samples, split):
    for index, sample in enumerate(
        samples,
        start=1,
    ):
        image_path = sample["image"]
        label_path = sample["label"]
        prefix = sample["prefix"]

        destination_stem = (
            f"{prefix}_{split}_{index:04d}"
        )

        copy_pair(
            image_path=image_path,
            label_path=label_path,
            split=split,
            destination_stem=destination_stem,
        )


def write_data_yaml():
    dataset_path = (
        OUTPUT_DATASET
        .resolve()
        .as_posix()
    )

    yaml_content = f"""path: {dataset_path}

train: images/train
val: images/val

names:
  0: hex_bolt
"""

    (
        OUTPUT_DATASET
        / "data.yaml"
    ).write_text(
        yaml_content,
        encoding="utf-8",
    )


def count_files(folder, suffix=None):
    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
    ]

    if suffix:
        files = [
            path
            for path in files
            if path.suffix.lower() == suffix
        ]

    return len(files)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("PREPARING HEX-BOLT DATASET V2")
    print("=" * 60)

    required_folders = [
        ORIGINAL_DATASET / "images" / "train",
        ORIGINAL_DATASET / "images" / "val",
        ORIGINAL_DATASET / "labels" / "train",
        ORIGINAL_DATASET / "labels" / "val",
        FRONT_IMAGES,
        FRONT_LABELS,
        END_IMAGES,
        END_LABELS,
    ]

    for folder in required_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Required folder missing: {folder}"
            )

    create_output_structure()

    original_counts = copy_original_dataset()

    front_samples = load_new_samples(
        FRONT_IMAGES,
        FRONT_LABELS,
        "front_missing_2",
    )

    end_samples = load_new_samples(
        END_IMAGES,
        END_LABELS,
        "end_missing_1",
    )

    front_train, front_val = split_new_samples(
        front_samples
    )

    end_train, end_val = split_new_samples(
        end_samples
    )

    copy_new_samples(
        front_train,
        "train",
    )

    copy_new_samples(
        front_val,
        "val",
    )

    copy_new_samples(
        end_train,
        "train",
    )

    copy_new_samples(
        end_val,
        "val",
    )

    write_data_yaml()

    train_images = count_files(
        OUTPUT_DATASET / "images" / "train"
    )

    train_labels = count_files(
        OUTPUT_DATASET / "labels" / "train",
        ".txt",
    )

    val_images = count_files(
        OUTPUT_DATASET / "images" / "val"
    )

    val_labels = count_files(
        OUTPUT_DATASET / "labels" / "val",
        ".txt",
    )

    if train_images != train_labels:
        raise RuntimeError(
            "Training image and label counts do not match."
        )

    if val_images != val_labels:
        raise RuntimeError(
            "Validation image and label counts do not match."
        )

    print()
    print("Original dataset copied:")
    print(
        f"  Train: {original_counts['train']}"
    )
    print(
        f"  Val:   {original_counts['val']}"
    )

    print()
    print("New Front Missing data:")
    print(
        f"  Train: {len(front_train)}"
    )
    print(
        f"  Val:   {len(front_val)}"
    )

    print()
    print("New End Missing data:")
    print(
        f"  Train: {len(end_train)}"
    )
    print(
        f"  Val:   {len(end_val)}"
    )

    print()
    print("Final V2 dataset:")
    print(
        f"  Train images: {train_images}"
    )
    print(
        f"  Train labels: {train_labels}"
    )
    print(
        f"  Val images:   {val_images}"
    )
    print(
        f"  Val labels:   {val_labels}"
    )
    print(
        f"  Total images: {train_images + val_images}"
    )

    print()
    print(
        f"Output folder: {OUTPUT_DATASET}"
    )
    print(
        f"Data YAML: {OUTPUT_DATASET / 'data.yaml'}"
    )
    print()
    print("DATASET V2 PREPARED SUCCESSFULLY")


if __name__ == "__main__":
    main()
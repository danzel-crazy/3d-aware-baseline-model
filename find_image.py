import os
import shutil
import json
from tqdm import tqdm


def collect_skull_images(base_folder, json_file, target_filenames=["2.png", "1.png"], output_folder="selected"):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Filter entries with category 'skull'
    skull_uids = [uid for uid, details in data.items() if details.get("category") == "skull"]

    # Create the output folder to collect images
    collected_folder = os.path.join(output_folder, os.path.basename(base_folder) + '_collected_images')
    os.makedirs(collected_folder, exist_ok=True)

    # Traverse the folder structure
    for root, dirs, files in tqdm(os.walk(base_folder), desc="Processing folders"):
        if "metadata.json" in files:
            metadata_path = os.path.join(root, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Extract UIDs from object_data and check if they are in skull_uids
            object_data = metadata.get("object_data", [])
            if any(obj.get("uid") in skull_uids for obj in object_data):
                for filename in target_filenames:
                    original_image_path = os.path.join(root, filename)
                    if os.path.exists(original_image_path):
                        new_filename = f"{os.path.basename(root)}_{filename}"
                        save_path = os.path.join(collected_folder, new_filename)
                        shutil.copy(original_image_path, save_path)


# Base directory path
base_folder = "DATASET/rotate/train"

# Path to the JSON file
json_file = "objaverse_cat_descriptions_64k.json"

# # Collect images with 'category': 'skull' from 'seen' and 'unseen' directories
# collect_skull_images(os.path.join(base_folder, "seen"), json_file)
# collect_skull_images(os.path.join(base_folder, "unseen"), json_file)

# Collect images with 'category': 'skull' from the base directory
collect_skull_images(base_folder, json_file)

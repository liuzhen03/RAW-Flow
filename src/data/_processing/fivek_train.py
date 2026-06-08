import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import cv2
import exiftool
import numpy as np
import rawpy
from sklearn.model_selection import train_test_split

from .patch_utils import patch_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("--num_patches", type=int, default=3)
parser.add_argument("--dividable_factor", type=int, default=32)
parser.add_argument("--overlap", default=False, action="store_true")
parser.add_argument("--skip_train", default=False, action="store_true")
parser.add_argument("--rgb_size", type=str, default="2X", choices=["1X", "2X"],
                    help="RGB resolution: 1X=half resolution, 2X=original resolution")

args = parser.parse_args()

raw_source_root = "data/fivek_dataset"
save_folder_root = "data/fivek_dataset_processed/"

model_names = ["NIKON D700", "Canon EOS 5D"]
raw_file_extensions = [".dng"]

test_ratio = 0.15
seed = 2817
num_processes = 16
num_patches = args.num_patches
dividable_factor = args.dividable_factor
overlap = args.overlap
rgb_size = args.rgb_size
dry_run = False
debug = False

dataset_name = "fivek"

if num_patches > 1:
    dataset_name += f"_patches_{num_patches}"

if overlap:
    dataset_name += "_overlap"

dataset_name += "_complete_dataset"

raw_source_root = os.path.expanduser(raw_source_root)
save_folder_root = os.path.join(save_folder_root, dataset_name)

raw_files_per_camera = defaultdict(list)

def get_camera_model_for_file(raw_path):
    raw_obj = SimpleNamespace(raw_path=raw_path)
    EXIFTOOL_SCRIPT_PATH = os.environ.get("EXIFTOOL", "exiftool")
    with exiftool.ExifToolHelper(executable=EXIFTOOL_SCRIPT_PATH) as et:
        metadata = et.get_metadata(raw_path)
        raw_camera_model = metadata[0]["EXIF:Model"]
        orientation = metadata[0]["XMP:Orientation"]
        raw_obj.camera_model = raw_camera_model
        raw_obj.orientation = orientation

    return raw_obj

print("[fivek/train] loading metadata from RAW files")
all_raw_files = []
for directory_path, _, files in os.walk(raw_source_root):
    for raw_file in files:
        file_ext = os.path.splitext(raw_file)[1]
        if file_ext in raw_file_extensions:
            raw_path = os.path.join(directory_path, raw_file)
            all_raw_files.append(raw_path)

with Pool(processes=32) as pool:
    raw_objects = pool.map(get_camera_model_for_file, all_raw_files)

for raw_object in raw_objects:
    raw_files_per_camera[raw_object.camera_model].append(raw_object)

def process_raw_file(raw_object, raw_destination_path, rgb_destination_path):
    raw_path = raw_object.raw_path
    orientation = raw_object.orientation
    camera_name = raw_object.camera_model

    file_name = os.path.splitext(os.path.basename(raw_path))[0]


    raw_file = rawpy.imread(raw_path)

    rgb_img = raw_file.postprocess()
    h, w, c = rgb_img.shape
    bayer_image = raw_file.raw_image_visible

    rgb_img_2x = rgb_img.astype(np.uint8)

    rgb_img_1x = cv2.resize(
        rgb_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA
    ).astype(np.uint8)

    context_256 = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8)
    context_128 = cv2.resize(rgb_img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.uint8)

    r_raw = bayer_image[0::2, 0::2][:, :, np.newaxis]
    g1_raw = bayer_image[0::2, 1::2][:, :, np.newaxis]
    g2_raw = bayer_image[1::2, 0::2][:, :, np.newaxis]
    b_raw = bayer_image[1::2, 1::2][:, :, np.newaxis]
    rggb_img = np.concatenate([r_raw, g1_raw, g2_raw, b_raw], axis=-1)

    cwb = raw_file.camera_whitebalance

    if orientation == 8:
        rggb_img = np.rot90(rggb_img, k=1)

    elif orientation == 6:
        rggb_img = np.rot90(rggb_img, k=3)

    elif orientation == 3:
        rggb_img = np.rot90(rggb_img, k=2)

    if camera_name == "Canon EOS 5D":
        rggb_img = np.maximum(rggb_img - 127.0, 0)

    rggb_img = rggb_img.astype(np.float32)

    raw_destination_paths = []
    rgb_destination_paths = []

    full_raw_filename = f"{file_name}.npz"
    full_rgb_1x_filename = f"{file_name}_1X.npy"
    full_rgb_2x_filename = f"{file_name}_2X.npy"
    context_256_filename = f"{file_name}_context_256.npy"
    context_128_filename = f"{file_name}_context_128.npy"

    full_raw_path = os.path.join(raw_destination_path, full_raw_filename)
    full_rgb_1x_path = os.path.join(rgb_destination_path, full_rgb_1x_filename)
    full_rgb_2x_path = os.path.join(rgb_destination_path, full_rgb_2x_filename)
    context_256_path = os.path.join(rgb_destination_path, context_256_filename)
    context_128_path = os.path.join(rgb_destination_path, context_128_filename)

    if not dry_run:
        np.savez(full_raw_path, raw=rggb_img, cwb=cwb)
        np.save(full_rgb_1x_path, rgb_img_1x)
        np.save(full_rgb_2x_path, rgb_img_2x)
        np.save(context_256_path, context_256)
        np.save(context_128_path, context_128)

    raw_destination_paths.append(full_raw_path)
    rgb_destination_paths.append(full_rgb_1x_path)

    if num_patches > 0:
        img_height, img_width = rggb_img.shape[:2]
        overlap_factor = 2 if overlap else 1
        patch_coordinates_list = patch_coordinates(
            img_height,
            img_width,
            num_patches,
            dividable_factor=dividable_factor,
            overlap_factor=overlap_factor,
        )

        for index, (index_x, index_y, x_start, y_start, x_end, y_end) in enumerate(
            patch_coordinates_list
        ):

            rggb_img_patch = rggb_img[y_start:y_end, x_start:x_end]

            rgb_img_1x_patch = rgb_img_1x[y_start:y_end, x_start:x_end]

            rgb_x_start_2x = x_start * 2
            rgb_y_start_2x = y_start * 2
            rgb_x_end_2x = x_end * 2
            rgb_y_end_2x = y_end * 2

            rgb_img_2x_patch = rgb_img_2x[rgb_y_start_2x:rgb_y_end_2x, rgb_x_start_2x:rgb_x_end_2x]

            patch_raw_filename = f"{file_name}_{index}.npz"
            patch_rgb_1x_filename = f"{file_name}_{index}_1X.npy"
            patch_rgb_2x_filename = f"{file_name}_{index}_2X.npy"

            patch_raw_path = os.path.join(raw_destination_path, patch_raw_filename)
            patch_rgb_1x_path = os.path.join(rgb_destination_path, patch_rgb_1x_filename)
            patch_rgb_2x_path = os.path.join(rgb_destination_path, patch_rgb_2x_filename)

            if not dry_run:
                np.savez(patch_raw_path, raw=rggb_img_patch, cwb=cwb)
                np.save(patch_rgb_1x_path, rgb_img_1x_patch)
                np.save(patch_rgb_2x_path, rgb_img_2x_patch)

            raw_destination_paths.append(patch_raw_path)
            rgb_destination_paths.append(patch_rgb_1x_path)

            coordinate_filename = f"{file_name}_{index}_coordinates.txt"
            coordinate_output_path = os.path.join(
                raw_destination_path, coordinate_filename
            )

            norm_x_start = x_start / img_width
            norm_y_start = y_start / img_height
            norm_x_end = x_end / img_width
            norm_y_end = y_end / img_height

            if not dry_run:

                with open(coordinate_output_path, "w") as f:
                    f.write(f"normalized:{norm_x_start:.6f},{norm_y_start:.6f},{norm_x_end:.6f},{norm_y_end:.6f}\n")

                    f.write(f"raw:{x_start},{y_start},{x_end},{y_end}\n")
                    f.write(f"rgb_2x:{rgb_x_start_2x},{rgb_y_start_2x},{rgb_x_end_2x},{rgb_y_end_2x}\n")

    return raw_destination_paths, rgb_destination_paths

for camera_model in model_names:
    print(f"[fivek/train] processing camera={camera_model}")

    raw_files = raw_files_per_camera[camera_model]
    raw_files.sort(key=lambda x: x.raw_path)
    print(f"[fivek/train] {len(raw_files)} images")

    train_raw_files, test_raw_files = train_test_split(
        raw_files, test_size=test_ratio, random_state=seed
    )

    camera_folder_name = camera_model.replace(" ", "_")

    files_per_split = {"train": train_raw_files, "test": test_raw_files}

    for split_name in ("train", "test"):
        if split_name == "train" and args.skip_train:
            continue

        raw_files_split = files_per_split[split_name]
        print(f"[fivek/train] {split_name}: {len(raw_files_split)} images")
        destination_folder_name_raw = f"{camera_folder_name}_{split_name}_raw"
        destination_folder_name_rgb = f"{camera_folder_name}_{split_name}_rgb"

        raw_destination_path = os.path.join(
            save_folder_root, destination_folder_name_raw
        )
        rgb_destination_path = os.path.join(
            save_folder_root, destination_folder_name_rgb
        )

        os.makedirs(raw_destination_path, exist_ok=True)
        os.makedirs(rgb_destination_path, exist_ok=True)

        raw_paths = []
        rgb_paths = []

        parallel = True
        if debug:
            raw_files_split = raw_files_split[:10]

        f = partial(
            process_raw_file,
            raw_destination_path=raw_destination_path,
            rgb_destination_path=rgb_destination_path,
        )

        if parallel:
            with Pool(processes=num_processes) as pool:
                results = pool.map(f, raw_files_split)
                raw_paths, rgb_paths = zip(*results)
        else:
            for raw_file in raw_files_split:
                raw_path, rgb_path = f(
                    raw_file,
                )
                raw_paths.append(raw_path)
                rgb_paths.append(rgb_path)

        raw_paths = [item for sublist in raw_paths for item in sublist]
        rgb_paths = [item for sublist in rgb_paths for item in sublist]

        all_files = []

        for raw_path, rgb_path in zip(raw_paths, rgb_paths):
            file_entry = {}
            raw_rel = os.path.relpath(raw_path, save_folder_root)
            rgb_rel = os.path.relpath(rgb_path, save_folder_root)

            file_entry["raw"] = raw_rel
            file_entry["rgb"] = rgb_rel

            base_name = os.path.basename(raw_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            is_patch = "_" in base_name_no_ext

            raw_dir = os.path.dirname(raw_path)
            rgb_dir = os.path.dirname(rgb_path)

            if is_patch:

                base_without_index = base_name_no_ext.rsplit("_", 1)[0]
                index = base_name_no_ext.rsplit("_", 1)[1]

                file_entry["raw_patch"] = raw_rel
                file_entry["rgb_1x_patch"] = rgb_rel

                rgb_2x_patch = os.path.join(rgb_dir, f"{base_without_index}_{index}_2X.npy")
                file_entry["rgb_2x_patch"] = os.path.relpath(rgb_2x_patch, save_folder_root)

                full_raw = os.path.join(raw_dir, f"{base_without_index}.npz")
                full_rgb_1x = os.path.join(rgb_dir, f"{base_without_index}_1X.npy")
                full_rgb_2x = os.path.join(rgb_dir, f"{base_without_index}_2X.npy")
                context_256 = os.path.join(rgb_dir, f"{base_without_index}_context_256.npy")
                context_128 = os.path.join(rgb_dir, f"{base_without_index}_context_128.npy")

                file_entry["full_raw"] = os.path.relpath(full_raw, save_folder_root)
                file_entry["full_rgb_1x"] = os.path.relpath(full_rgb_1x, save_folder_root)
                file_entry["full_rgb_2x"] = os.path.relpath(full_rgb_2x, save_folder_root)
                file_entry["context_256"] = os.path.relpath(context_256, save_folder_root)
                file_entry["context_128"] = os.path.relpath(context_128, save_folder_root)

                coord_file = os.path.join(raw_dir, f"{base_without_index}_{index}_coordinates.txt")
                file_entry["coordinates"] = os.path.relpath(coord_file, save_folder_root)
            else:

                file_entry["full_raw"] = raw_rel
                file_entry["full_rgb_1x"] = rgb_rel

                base_without_ext = os.path.splitext(base_name)[0]
                full_rgb_2x = os.path.join(rgb_dir, f"{base_without_ext}_2X.npy")
                context_256 = os.path.join(rgb_dir, f"{base_without_ext}_context_256.npy")
                context_128 = os.path.join(rgb_dir, f"{base_without_ext}_context_128.npy")

                file_entry["full_rgb_2x"] = os.path.relpath(full_rgb_2x, save_folder_root)
                file_entry["context_256"] = os.path.relpath(context_256, save_folder_root)
                file_entry["context_128"] = os.path.relpath(context_128, save_folder_root)

            all_files.append(file_entry)

        csv_path = os.path.join(
            save_folder_root, f"{camera_folder_name}_{split_name}.txt"
        )
        if not dry_run:
            with open(csv_path, "w") as f:
                for file_entry in all_files:
                    f.write(f"{file_entry['raw']},{file_entry['rgb']}\n")

        comprehensive_csv_path = os.path.join(
            save_folder_root, f"{camera_folder_name}_{split_name}_comprehensive.txt"
        )
        if not dry_run:
            with open(comprehensive_csv_path, "w") as f:

                f.write("raw,rgb,full_raw,full_rgb_1x,full_rgb_2x,context_256,context_128")
                if any("coordinates" in file_entry for file_entry in all_files):
                    f.write(",coordinates")
                f.write("\n")

                for file_entry in all_files:

                    f.write(f"{file_entry['raw']},{file_entry['rgb']}")

                    f.write(f",{file_entry['full_raw']},{file_entry['full_rgb_1x']},{file_entry['full_rgb_2x']}")

                    f.write(f",{file_entry['context_256']},{file_entry['context_128']}")

                    if "coordinates" in file_entry:
                        f.write(f",{file_entry['coordinates']}")

                    f.write("\n")

        norm_coords_path = os.path.join(
            save_folder_root, f"{camera_folder_name}_{split_name}_normalized_coords.txt"
        )
        if not dry_run:
            with open(norm_coords_path, "w") as f:

                f.write("image,patch_index,norm_x_start,norm_y_start,norm_x_end,norm_y_end\n")

                for file_entry in all_files:
                    if "coordinates" in file_entry:

                        coord_path = file_entry["coordinates"]
                        base_name = os.path.basename(coord_path)

                        if "_coordinates.txt" in base_name:
                            parts = base_name.replace("_coordinates.txt", "").split("_")
                            image_name = "_".join(parts[:-1])
                            patch_index = parts[-1]

                            coord_full_path = os.path.join(save_folder_root, coord_path)
                            if os.path.exists(coord_full_path):
                                with open(coord_full_path, "r") as coord_file:
                                    first_line = coord_file.readline().strip()
                                    if first_line.startswith("normalized:"):
                                        coords = first_line.replace("normalized:", "")
                                        f.write(f"{image_name},{patch_index},{coords}\n")

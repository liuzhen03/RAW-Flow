import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import exiftool
import numpy as np
import rawpy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .patch_utils import patch_coordinates
except ModuleNotFoundError:

    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    from .patch_utils import patch_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("--num_patches", type=int, default=1)
parser.add_argument("--dividable_factor", type=int, default=32)
parser.add_argument("--overlap", default=False, action="store_true")
parser.add_argument("--skip_train", default=False, action="store_true")
parser.add_argument("--dry_run", default=False, action="store_true",
                    help="dry run - print only, do not write")

args = parser.parse_args()

raw_source_root = "data/PASCALRAW/original/raw"
trainval_root = "data/PASCALRAW/trainval"
save_folder_root = "data/PASCALRAW_processed/"
file_split_root = os.environ.get("PASCALRAW_SPLITS", "data/PASCALRAW_splits")

raw_file_extensions = [".nef"]

num_processes = 16
num_patches = args.num_patches
dividable_factor = args.dividable_factor
overlap = args.overlap
dry_run = args.dry_run
debug = False

dataset_name = "pascalraw"

if num_patches > 1:
    dataset_name += f"_patches_{num_patches}"

if overlap:
    dataset_name += "_overlap"

raw_source_root = os.path.expanduser(raw_source_root)
save_folder_root = os.path.join(save_folder_root, dataset_name)

def load_split_ids(split_file):
    ids = []
    with open(split_file, 'r') as f:
        for line in f:
            ids.append(line.strip())
    return ids

def process_raw_file(raw_path, raw_destination_path, rgb_destination_path):
    file_name = os.path.splitext(os.path.basename(raw_path))[0]


    try:

        raw_file = rawpy.imread(raw_path)

        EXIFTOOL_SCRIPT_PATH = os.environ.get("EXIFTOOL", "exiftool")
        orientation = 1
        camera_name = "Unknown"

        try:
            with exiftool.ExifToolHelper(executable=EXIFTOOL_SCRIPT_PATH) as et:
                metadata = et.get_metadata(raw_path)
                if "XMP:Orientation" in metadata[0]:
                    orientation = metadata[0]["XMP:Orientation"]
                elif "EXIF:Orientation" in metadata[0]:
                    orientation = metadata[0]["EXIF:Orientation"]
                if "EXIF:Model" in metadata[0]:
                    camera_name = metadata[0]["EXIF:Model"]
        except Exception as e:
            print(f"Warning: Could not read EXIF data for {raw_path}: {e}")

        rgb_img = raw_file.postprocess()
        h, w, c = rgb_img.shape

        bayer_image = raw_file.raw_image_visible

        rgb_img_full = rgb_img.astype(np.uint8)

        r_raw = bayer_image[0::2, 0::2][:, :, np.newaxis]
        g1_raw = bayer_image[0::2, 1::2][:, :, np.newaxis]
        g2_raw = bayer_image[1::2, 0::2][:, :, np.newaxis]
        b_raw = bayer_image[1::2, 1::2][:, :, np.newaxis]
        rggb_img = np.concatenate([r_raw, g1_raw, g2_raw, b_raw], axis=-1)

        cwb = raw_file.camera_whitebalance

        if orientation == 8:
            rggb_img = np.rot90(rggb_img, k=1)
            rgb_img_full = np.rot90(rgb_img_full, k=1)
        elif orientation == 6:
            rggb_img = np.rot90(rggb_img, k=3)
            rgb_img_full = np.rot90(rgb_img_full, k=3)
        elif orientation == 3:
            rggb_img = np.rot90(rggb_img, k=2)
            rgb_img_full = np.rot90(rgb_img_full, k=2)

        if camera_name == "Canon EOS 5D":
            rggb_img = np.maximum(rggb_img - 127.0, 0)

        rggb_img = rggb_img.astype(np.float32)

        raw_destination_paths = []
        rgb_destination_paths = []

        if num_patches > 1:

            raw_height, raw_width = rggb_img.shape[:2]
            overlap_factor = 2 if overlap else 1

            patch_coordinates_list = patch_coordinates(
                raw_height,
                raw_width,
                num_patches,
                dividable_factor=dividable_factor,
                overlap_factor=overlap_factor,
            )

            for index, (index_x, index_y, x_start, y_start, x_end, y_end) in enumerate(
                patch_coordinates_list
            ):

                rggb_img_patch = rggb_img[y_start:y_end, x_start:x_end]

                rgb_x_start = x_start * 2
                rgb_y_start = y_start * 2
                rgb_x_end = x_end * 2
                rgb_y_end = y_end * 2
                rgb_img_patch = rgb_img_full[rgb_y_start:rgb_y_end, rgb_x_start:rgb_x_end]

                raw_file_name = f"{file_name}_{index}.npz"
                rgb_file_name = f"{file_name}_{index}.npy"

                raw_np_path = os.path.join(raw_destination_path, raw_file_name)
                rgb_np_path = os.path.join(rgb_destination_path, rgb_file_name)

                if not dry_run:
                    os.makedirs(raw_destination_path, exist_ok=True)
                    os.makedirs(rgb_destination_path, exist_ok=True)
                    np.savez(raw_np_path, raw=rggb_img_patch, cwb=cwb)
                    np.save(rgb_np_path, rgb_img_patch)

                raw_destination_paths.append(raw_np_path)
                rgb_destination_paths.append(rgb_np_path)

                if not dry_run:
                    coordinate_filename = f"{file_name}_{index}.txt"
                    coordinate_output_path = os.path.join(
                        raw_destination_path, coordinate_filename
                    )
                    with open(coordinate_output_path, "w") as f:
                        f.write(f"{x_start},{y_start},{x_end},{y_end}")
        else:
            raw_file_name = file_name + ".npz"
            rgb_file_name = file_name + ".npy"

            raw_np_path = os.path.join(raw_destination_path, raw_file_name)
            rgb_np_path = os.path.join(rgb_destination_path, rgb_file_name)

            if not dry_run:
                os.makedirs(raw_destination_path, exist_ok=True)
                os.makedirs(rgb_destination_path, exist_ok=True)
                np.savez(raw_np_path, raw=rggb_img, cwb=cwb)
                np.save(rgb_np_path, rgb_img_full)

            raw_destination_paths.append(raw_np_path)
            rgb_destination_paths.append(rgb_np_path)

        raw_file.close()
        return raw_destination_paths, rgb_destination_paths

    except Exception as e:
        print(f"Error processing {raw_path}: {str(e)}")
        return [], []

def main():
    print("[pascalraw/eval] processing")

    if not os.path.exists(raw_source_root):
        print(f"[pascalraw/eval] raw root: {raw_source_root}")
        return

    if not os.path.exists(file_split_root):
        print(f"[pascalraw/eval] splits root: {file_split_root}")
        return

    splits = ["train", "test"]

    for split_name in splits:
        if split_name == "train" and args.skip_train:
            continue

        output_split_name = split_name

        print(f"[pascalraw/eval] {split_name} -> {output_split_name}")

        split_file = os.path.join(file_split_root, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"[pascalraw/eval] split file not found: {split_file}")
            continue

        image_ids = load_split_ids(split_file)
        print(f"[pascalraw/eval] {split_name}: {len(image_ids)} ids")

        destination_folder_name_raw = f"PASCALRAW_{output_split_name}_raw"
        destination_folder_name_rgb = f"PASCALRAW_{output_split_name}_rgb"

        raw_destination_path = os.path.join(save_folder_root, destination_folder_name_raw)
        rgb_destination_path = os.path.join(save_folder_root, destination_folder_name_rgb)

        if not dry_run:
            os.makedirs(raw_destination_path, exist_ok=True)
            os.makedirs(rgb_destination_path, exist_ok=True)

        raw_files = []
        missing_files = []

        for image_id in image_ids:
            raw_path = os.path.join(raw_source_root, f"{image_id}.nef")
            if os.path.exists(raw_path):
                raw_files.append(raw_path)
            else:
                missing_files.append(raw_path)

        if missing_files:
            print(f"[pascalraw/eval] missing {len(missing_files)} RAW files")

        print(f"[pascalraw/eval] {len(raw_files)} RAW files")

        if debug:
            raw_files = raw_files[:10]

        raw_paths_all = []
        rgb_paths_all = []

        f = partial(
            process_raw_file,
            raw_destination_path=raw_destination_path,
            rgb_destination_path=rgb_destination_path,
        )

        parallel = True
        if parallel and len(raw_files) > 1:
            with Pool(processes=num_processes) as pool:
                results = pool.map(f, raw_files)
        else:
            results = []
            for raw_file_path in raw_files:
                result = f(raw_file_path)
                results.append(result)

        for raw_paths, rgb_paths in results:
            raw_paths_all.extend(raw_paths)
            rgb_paths_all.extend(rgb_paths)

        print(f"[pascalraw/eval] wrote {len(raw_paths_all)} files")

        if not dry_run and raw_paths_all:
            raw_paths_rel = [
                os.path.relpath(raw_path, save_folder_root) for raw_path in raw_paths_all
            ]
            rgb_paths_rel = [
                os.path.relpath(rgb_path, save_folder_root) for rgb_path in rgb_paths_all
            ]

            csv_path = os.path.join(save_folder_root, f"PASCALRAW_{output_split_name}.txt")
            with open(csv_path, "w") as f:
                for raw_path, rgb_path in zip(raw_paths_rel, rgb_paths_rel):
                    f.write(f"{raw_path},{rgb_path}\n")


    print("[pascalraw/eval] done")

if __name__ == "__main__":
    main()

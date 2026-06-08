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
parser.add_argument("--num_patches", type=int, default=3)
parser.add_argument("--dividable_factor", type=int, default=32)
parser.add_argument("--overlap", default=False, action="store_true")
parser.add_argument("--skip_train", default=False, action="store_true")
parser.add_argument("--dry_run", default=False, action="store_true",
                    help="dry run - print only, do not write")
parser.add_argument("--rgb_size", type=str, default="2X", choices=["1X", "2X"],
                    help="RGB resolution: 1X (half) or 2X (full)")
parser.add_argument("--save_full_image", default=False, action="store_true",
                    help="save full images alongside patches")
parser.add_argument("--context_size", type=int, default=256,
                    help="context RGB size (default 256)")
parser.add_argument("--save_coordinates", default=False, action="store_true",
                    help="save patch coordinates")

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
rgb_size = args.rgb_size
save_full_image = args.save_full_image
context_size = args.context_size
save_coordinates = args.save_coordinates
debug = False

dataset_name = "pascalraw"

if num_patches > 1:
    dataset_name += f"_patches_{num_patches}"

if overlap:
    dataset_name += "_overlap"

if rgb_size == "1X":
    dataset_name += "_1X"

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

        rgb_img_2x = rgb_img.astype(np.uint8)

        rgb_img_1x = cv2.resize(
            rgb_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA
        ).astype(np.uint8)

        context_img = cv2.resize(rgb_img, (context_size, context_size), interpolation=cv2.INTER_AREA).astype(np.uint8)

        r_raw = bayer_image[0::2, 0::2][:, :, np.newaxis]
        g1_raw = bayer_image[0::2, 1::2][:, :, np.newaxis]
        g2_raw = bayer_image[1::2, 0::2][:, :, np.newaxis]
        b_raw = bayer_image[1::2, 1::2][:, :, np.newaxis]
        rggb_img = np.concatenate([r_raw, g1_raw, g2_raw, b_raw], axis=-1)

        cwb = raw_file.camera_whitebalance

        if orientation == 8:
            rggb_img = np.rot90(rggb_img, k=1)
            rgb_img_1x = np.rot90(rgb_img_1x, k=1)
            rgb_img_2x = np.rot90(rgb_img_2x, k=1)
            context_img = np.rot90(context_img, k=1)
        elif orientation == 6:
            rggb_img = np.rot90(rggb_img, k=3)
            rgb_img_1x = np.rot90(rgb_img_1x, k=3)
            rgb_img_2x = np.rot90(rgb_img_2x, k=3)
            context_img = np.rot90(context_img, k=3)
        elif orientation == 3:
            rggb_img = np.rot90(rggb_img, k=2)
            rgb_img_1x = np.rot90(rgb_img_1x, k=2)
            rgb_img_2x = np.rot90(rgb_img_2x, k=2)
            context_img = np.rot90(context_img, k=2)

        if camera_name == "Canon EOS 5D":
            rggb_img = np.maximum(rggb_img - 127.0, 0)

        rggb_img = rggb_img.astype(np.float32)

        raw_destination_paths = []
        rgb_destination_paths = []

        if save_full_image:
            full_raw_filename = f"{file_name}.npz"
            context_filename = f"{file_name}_context_{context_size}.npy"

            full_raw_path = os.path.join(raw_destination_path, full_raw_filename)
            context_path = os.path.join(rgb_destination_path, context_filename)

            if rgb_size == "2X":
                full_rgb_filename = f"{file_name}_2X.npy"
                full_rgb_path = os.path.join(rgb_destination_path, full_rgb_filename)
                full_rgb_img = rgb_img_2x
            else:
                full_rgb_filename = f"{file_name}_1X.npy"
                full_rgb_path = os.path.join(rgb_destination_path, full_rgb_filename)
                full_rgb_img = rgb_img_1x

            if not dry_run:
                os.makedirs(raw_destination_path, exist_ok=True)
                os.makedirs(rgb_destination_path, exist_ok=True)
                np.savez(full_raw_path, raw=rggb_img, cwb=cwb)
                np.save(full_rgb_path, full_rgb_img)
                np.save(context_path, context_img)

            raw_destination_paths.append(full_raw_path)
            rgb_destination_paths.append(full_rgb_path)
        else:

            if not dry_run:
                context_filename = f"{file_name}_context_{context_size}.npy"
                context_path = os.path.join(rgb_destination_path, context_filename)
                os.makedirs(rgb_destination_path, exist_ok=True)
                np.save(context_path, context_img)

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

                if rgb_size == "2X":

                    rgb_x_start = x_start * 2
                    rgb_y_start = y_start * 2
                    rgb_x_end = x_end * 2
                    rgb_y_end = y_end * 2
                    rgb_img_patch = rgb_img_2x[rgb_y_start:rgb_y_end, rgb_x_start:rgb_x_end]
                    patch_rgb_filename = f"{file_name}_{index}_2X.npy"
                else:

                    rgb_img_patch = rgb_img_1x[y_start:y_end, x_start:x_end]
                    patch_rgb_filename = f"{file_name}_{index}_1X.npy"

                patch_raw_filename = f"{file_name}_{index}.npz"

                patch_raw_path = os.path.join(raw_destination_path, patch_raw_filename)
                patch_rgb_path = os.path.join(rgb_destination_path, patch_rgb_filename)

                if not dry_run:
                    os.makedirs(raw_destination_path, exist_ok=True)
                    os.makedirs(rgb_destination_path, exist_ok=True)
                    np.savez(patch_raw_path, raw=rggb_img_patch, cwb=cwb)
                    np.save(patch_rgb_path, rgb_img_patch)

                raw_destination_paths.append(patch_raw_path)
                rgb_destination_paths.append(patch_rgb_path)

                if save_coordinates:
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
                            if rgb_size == "2X":
                                f.write(f"rgb_2x:{rgb_x_start},{rgb_y_start},{rgb_x_end},{rgb_y_end}\n")
                            else:
                                f.write(f"rgb_1x:{x_start},{y_start},{x_end},{y_end}\n")

        raw_file.close()
        return raw_destination_paths, rgb_destination_paths

    except Exception as e:
        print(f"Error processing {raw_path}: {str(e)}")
        return [], []

def main():
    print("[pascalraw/train] processing")

    if not os.path.exists(raw_source_root):
        print(f"[pascalraw/train] raw root: {raw_source_root}")
        return

    splits = ["train", "test"]

    for split_name in splits:
        if split_name == "train" and args.skip_train:
            continue

        output_split_name = split_name

        print(f"[pascalraw/train] {split_name} -> {output_split_name}")

        split_file = os.path.join(file_split_root, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"[pascalraw/train] split file not found: {split_file}")
            continue

        image_ids = load_split_ids(split_file)
        print(f"[pascalraw/train] {split_name}: {len(image_ids)} ids")

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
            print(f"[pascalraw/train] missing {len(missing_files)} RAW files")

        print(f"[pascalraw/train] {len(raw_files)} RAW files")

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

        print(f"[pascalraw/train] wrote {len(raw_paths_all)} patches")

        if not dry_run and raw_paths_all:
            all_files = []

            for raw_path, rgb_path in zip(raw_paths_all, rgb_paths_all):
                file_entry = {}
                raw_rel = os.path.relpath(raw_path, save_folder_root)
                rgb_rel = os.path.relpath(rgb_path, save_folder_root)

                file_entry["raw"] = raw_rel
                file_entry["rgb"] = rgb_rel

                base_name = os.path.basename(raw_path)
                base_name_no_ext = os.path.splitext(base_name)[0]
                is_patch = "_" in base_name_no_ext and base_name_no_ext.split("_")[-1].isdigit()

                raw_dir = os.path.dirname(raw_path)
                rgb_dir = os.path.dirname(rgb_path)

                if is_patch:

                    base_without_index = base_name_no_ext.rsplit("_", 1)[0]
                    index = base_name_no_ext.rsplit("_", 1)[1]

                    context_file = os.path.join(rgb_dir, f"{base_without_index}_context_{context_size}.npy")
                    file_entry[f"context_{context_size}"] = os.path.relpath(context_file, save_folder_root)

                    if save_full_image:
                        full_raw = os.path.join(raw_dir, f"{base_without_index}.npz")
                        full_rgb = os.path.join(rgb_dir, f"{base_without_index}.npy")
                        file_entry["full_raw"] = os.path.relpath(full_raw, save_folder_root)
                        file_entry["full_rgb"] = os.path.relpath(full_rgb, save_folder_root)

                    if save_coordinates:
                        coord_file = os.path.join(raw_dir, f"{base_without_index}_{index}_coordinates.txt")
                        file_entry["coordinates"] = os.path.relpath(coord_file, save_folder_root)
                else:

                    base_without_ext = os.path.splitext(base_name)[0]
                    context_file = os.path.join(rgb_dir, f"{base_without_ext}_context_{context_size}.npy")
                    file_entry[f"context_{context_size}"] = os.path.relpath(context_file, save_folder_root)

                all_files.append(file_entry)

            csv_path = os.path.join(save_folder_root, f"PASCALRAW_{output_split_name}.txt")
            with open(csv_path, "w") as f:
                for file_entry in all_files:
                    f.write(f"{file_entry['raw']},{file_entry['rgb']}\n")

            comprehensive_csv_path = os.path.join(
                save_folder_root, f"PASCALRAW_{output_split_name}_comprehensive.txt"
            )
            with open(comprehensive_csv_path, "w") as f:
                headers = ["raw", "rgb", f"context_{context_size}"]
                if save_full_image:
                    headers.extend(["full_raw", "full_rgb"])
                if save_coordinates:
                    headers.append("coordinates")
                f.write(",".join(headers) + "\n")

                for file_entry in all_files:
                    row = [
                        file_entry["raw"],
                        file_entry["rgb"],
                        file_entry.get(f"context_{context_size}", "")
                    ]

                    if save_full_image:
                        row.extend([
                            file_entry.get("full_raw", ""),
                            file_entry.get("full_rgb", "")
                        ])

                    if save_coordinates:
                        row.append(file_entry.get("coordinates", ""))

                    f.write(",".join(row) + "\n")


    print("[pascalraw/train] done")

if __name__ == "__main__":
    main()

import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def load_all_tensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Merge a VACE-only training checkpoint into a full Wan VACE base checkpoint.")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--vace_ckpt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--base_prefix", default="vace.")
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_tensors = load_all_tensors(args.base_ckpt)
    vace_tensors = load_all_tensors(args.vace_ckpt)

    matched = []
    missing = []
    shape_mismatch = []

    for key, tensor in vace_tensors.items():
        target_key = f"{args.base_prefix}{key}"
        if target_key not in base_tensors:
            missing.append(key)
            continue
        if tuple(base_tensors[target_key].shape) != tuple(tensor.shape):
            shape_mismatch.append((key, tuple(tensor.shape), tuple(base_tensors[target_key].shape)))
            continue
        matched.append((key, target_key))

    report = {
        "base_total": len(base_tensors),
        "vace_patch_total": len(vace_tensors),
        "matched": len(matched),
        "missing": len(missing),
        "shape_mismatch": len(shape_mismatch),
        "missing_sample": missing[:20],
        "shape_mismatch_sample": shape_mismatch[:10],
    }
    print(f"base_total={report['base_total']}")
    print(f"vace_patch_total={report['vace_patch_total']}")
    print(f"matched={report['matched']}")
    print(f"missing={report['missing']}")
    print(f"shape_mismatch={report['shape_mismatch']}")
    if missing:
        print("missing_sample=", report["missing_sample"])
    if shape_mismatch:
        print("shape_mismatch_sample=", report["shape_mismatch_sample"])
    if args.report_json is not None:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if missing or shape_mismatch:
        raise RuntimeError("Checkpoint merge aborted because some VACE keys are missing or mismatched.")

    if args.dry_run:
        print("dry_run=True, no output file written.")
        return

    merged = dict(base_tensors)
    for src_key, dst_key in matched:
        merged[dst_key] = vace_tensors[src_key]

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    save_file(merged, args.output_ckpt)
    print(f"saved={args.output_ckpt}")


if __name__ == "__main__":
    main()

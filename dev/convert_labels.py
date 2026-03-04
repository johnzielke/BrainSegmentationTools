from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "dev" / "freesurfer" / "mri_synthseg"
DEFAULT_OUTPUT = REPO_ROOT / "brain_segmentation_tools" / "constants"

VALID_FIELDS = {"labels", "names", "classes"}
V2_ONLY_CATEGORIES = {"parcellation", "denoiser"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SynthSeg label/class .npy files from the FreeSurfer submodule "
            "into Python constants modules under brain_segmentation_tools/constants."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input directory containing synthseg*.npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output package directory for constants modules.",
    )
    return parser.parse_args()


def parse_file_name(stem: str) -> tuple[str, str, str] | None:
    version = None
    base = stem
    if stem.endswith("_2.0"):
        version = "v2_0"
        base = stem.removesuffix("_2.0")

    parts = base.split("_")
    if len(parts) != 3 or parts[0] != "synthseg":
        return None

    _, category, field = parts
    if field not in VALID_FIELDS:
        return None

    if version is None:
        version = "v2_0" if category in V2_ONLY_CATEGORIES else "v1_0"

    return version, category, field


def write_constants_modules(*, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    versions: dict[str, dict[str, dict[str, list[int] | list[str]]]] = {
        "v1_0": {},
        "v2_0": {},
    }
    skipped: list[str] = []

    for file_path in sorted(input_dir.glob("synthseg*.npy")):
        parsed = parse_file_name(file_path.stem)
        if parsed is None:
            skipped.append(file_path.name)
            continue

        version, category, field = parsed
        data = np.load(file_path)
        versions.setdefault(version, {}).setdefault(category, {})[field] = data.tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").write_text("")

    for version, version_data in versions.items():
        version_dir = output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "__init__.py").write_text("")

        category_files = set(version_data)
        for existing in version_dir.glob("*.py"):
            if existing.stem != "__init__" and existing.stem not in category_files:
                existing.unlink()

        for category, category_data in sorted(version_data.items()):
            content = "RESOURCE = " + pformat(category_data, width=88, sort_dicts=False)
            (version_dir / f"{category}.py").write_text(content + "\n")

    print(f"Wrote constants modules to: {output_dir}")
    if skipped:
        print("Skipped files:")
        for name in skipped:
            print(f"  - {name}")


def main() -> None:
    args = parse_args()
    write_constants_modules(input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()

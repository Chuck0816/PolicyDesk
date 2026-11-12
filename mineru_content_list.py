"""Write a content list with MinerU 4.x for PolicyDesk ingestion.

MinerU 4.x no longer has the `mineru -p <pdf> -o <dir>` command that wrote
`<name>_content_list.json`. This script restores that output through MinerU's Python API.
PolicyDesk runs it with the Python of MinerU's own environment:

    /path/to/mineru-env/bin/python mineru_content_list.py -p policy.pdf -o output_dir

It writes `output_dir/<pdf stem>_content_list.json` plus the images that file references.
It imports only MinerU and the standard library, never PolicyDesk modules.
Set MINERU_TIER (flash, basic, standard, advanced) to change the parse tier.
"""

import argparse
import json
import os
from pathlib import Path


def convert(pdf, output_dir, tier):
    from docvortex.export import materialize_middle, validate_materialized_assets
    from mineru.parser import parse
    from mineru.render import render_content_list

    result = parse(pdf, tier=tier, page_range="")  # An empty range parses every page.
    middle_json, assets = materialize_middle(result.middle_json)
    validate_materialized_assets(middle_json, assets)

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative, payload in assets.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"MinerU asset path leaves the output folder: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    path = root / f"{pdf.stem}_content_list.json"
    items = render_content_list(middle_json)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Write a MinerU content list for PolicyDesk.")
    parser.add_argument("-p", "--pdf", required=True, help="PDF to parse")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    args = parser.parse_args()
    tier = os.getenv("MINERU_TIER", "standard")
    print(convert(Path(args.pdf), Path(args.output), tier))


# MinerU starts worker processes that re-import this file, so work only under this guard.
if __name__ == "__main__":
    main()

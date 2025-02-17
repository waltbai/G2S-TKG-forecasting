"""
Convert dataset from RE-GCN to standardized format.
"""
import argparse
import os

from yaml import load, Loader


def convert_icews14(input_dir: str, output_dir: str):
    """
    Convert ICEWS14 dataset to standardized format.
    """
    # Process entity2id: replace "_" with " "
    in_path = os.path.join(input_dir, "entity2id.txt")
    out_path = os.path.join(output_dir, "entity2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process relation2id: replace "_" with " "
    in_path = os.path.join(input_dir, "relation2id.txt")
    out_path = os.path.join(output_dir, "relation2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process fact files:
    #   In original ICEWS14, time starts with 1
    for part in ["train", "valid", "test"]:
        in_path = os.path.join(input_dir, f"{part}.txt")
        out_path = os.path.join(output_dir, f"{part}.txt")
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s, p, o, t = line.strip().split("\t")
                t = int(t) - 1
                fout.write(f"{s}\t{p}\t{o}\t{t}\n")


def convert_icews18(input_dir: str, output_dir: str):
    """
    Convert ICEWS18 dataset to standardized format.
    """
    # Process entity2id: replace "_" with " "
    in_path = os.path.join(input_dir, "entity2id.txt")
    out_path = os.path.join(output_dir, "entity2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process relation2id: replace "_" with " "
    in_path = os.path.join(input_dir, "relation2id.txt")
    out_path = os.path.join(output_dir, "relation2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process fact files:
    #   In original ICEWS18, time increases by 24 hours,
    #   there exists redundant spaces,
    #   and there exists an additional column
    for part in ["train", "valid", "test"]:
        in_path = os.path.join(input_dir, f"{part}.txt")
        out_path = os.path.join(output_dir, f"{part}.txt")
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s, p, o, t = line.strip().split("\t")[:4]
                t = int(t) // 24
                fout.write(f"{s.strip()}\t{p.strip()}\t{o.strip()}\t{t}\n")


def convert_icews05_15(input_dir: str, output_dir: str):
    """
    Convert ICEWS05-15 dataset to standardized format.
    """
    # Process entity2id: replace "_" with " "
    in_path = os.path.join(input_dir, "entity2id.txt")
    out_path = os.path.join(output_dir, "entity2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process relation2id: replace "_" with " "
    in_path = os.path.join(input_dir, "relation2id.txt")
    out_path = os.path.join(output_dir, "relation2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process fact files: copy original files
    for part in ["train", "valid", "test"]:
        in_path = os.path.join(input_dir, f"{part}.txt")
        out_path = os.path.join(output_dir, f"{part}.txt")
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s, p, o, t = line.strip().split("\t")
                fout.write(f"{s}\t{p}\t{o}\t{t}\n")


def convert_gdelt(input_dir: str, output_dir: str, cameo_path: str):
    """
    Convert GDELT dataset to standardized format.
    """
    # Process entity2id: remove brackets and use title string format
    in_path = os.path.join(input_dir, "entity2id.txt")
    out_path = os.path.join(output_dir, "entity2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            # In GDELT, the brackets always at the end of an entity name
            try:
                name = name[:name.index("(")].strip()
            except ValueError:
                pass
            name = name.title()
            fout.write(f"{name}\t{id}\n")

    # Process relation2id: convert CAMEO event code to relation name
    in_path = os.path.join(input_dir, "relation2id.txt")
    out_path = os.path.join(output_dir, "relation2id.txt")
    with open(cameo_path, "r") as f:
        cameo = load(f, Loader=Loader)["event_code"]
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = cameo[name]
            fout.write(f"{name}\t{id}\n")

    # Process fact files:
    #   In original GDELT, time increases by 15 minutes,
    #   there exists redundant spaces,
    #   and there exists an additional column
    for part in ["train", "valid", "test"]:
        in_path = os.path.join(input_dir, f"{part}.txt")
        out_path = os.path.join(output_dir, f"{part}.txt")
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s, p, o, t = line.strip().split("\t")[:4]
                t = int(t) // 15
                fout.write(f"{s.strip()}\t{p.strip()}\t{o.strip()}\t{t}\n")


def convert_yago(input_dir: str, output_dir: str):
    """
    Convert YAGO dataset to standardized format.
    """
    # Process entity2id: remove brackets and use title string format
    in_path = os.path.join(input_dir, "entity2id.txt")
    out_path = os.path.join(output_dir, "entity2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")[:2]
            # In GDELT, the brackets always at the end of an entity name
            name = name.strip("<>").replace("_", " ")
            fout.write(f"{name}\t{id}\n")

    # Process relation2id: convert CAMEO event code to relation name
    in_path = os.path.join(input_dir, "relation2id.txt")
    out_path = os.path.join(output_dir, "relation2id.txt")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            name, id = line.strip().split("\t")
            name = name.strip("<>")
            fout.write(f"{name}\t{id}\n")

    # Process fact files:
    #   remove extra columns
    for part in ["train", "valid", "test"]:
        in_path = os.path.join(input_dir, f"{part}.txt")
        out_path = os.path.join(output_dir, f"{part}.txt")
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                s, p, o, t = line.strip().split("\t")[:4]
                t = int(t)
                fout.write(f"{s.strip()}\t{p.strip()}\t{o.strip()}\t{t}\n")


def guess_dataset(input_dir: str) -> str:
    """
    Guess dataset name by input directory.
    """
    dir_name = os.path.split(input_dir)[-1]
    if dir_name in ["ICEWS14", "ICEWS14s"]:
        guess_name = "ICEWS14"
    else:
        guess_name = dir_name.upper()
    return guess_name


def get_args():
    """
    Get arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--cameo_path", type=str, default="cameo.yaml")
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = get_args()
    if args.dataset is None:
        args.dataset = guess_dataset(args.input_dir)
    else:
        args.dataset = args.dataset.upper()

    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset == "ICEWS14":
        convert_icews14(input_dir, output_dir)
    elif args.dataset == "ICEWS18":
        convert_icews18(input_dir, output_dir)
    elif args.dataset == "ICEWS05-15":
        convert_icews05_15(input_dir, output_dir)
    elif args.dataset == "GDELT":
        convert_gdelt(input_dir, output_dir, args.cameo_path)
    elif args.dataset == "YAGO":
        convert_yago(input_dir, output_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()

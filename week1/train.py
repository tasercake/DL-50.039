#!/usr/bin/env python


if __name__ == "__main__":
    with concepts_file.open("r") as f:
        lines = [line.rstrip() for line in f.readlines()]
        concepts = {
            int(index): name for index, name in [line.split() for line in lines[1:]]
        }
        print(f"Discovered {len(concepts)} concepts")
    concepts_file = Path("concepts_2011.txt")

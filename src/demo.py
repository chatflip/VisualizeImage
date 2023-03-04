import argparse
import os

from Visualizer import Visualizer


def opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="sample.jpg")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = opt()
    os.makedirs(args.output_dir, exist_ok=True)
    visualizer = Visualizer(args.output_dir)
    visualizer(args.input_path)

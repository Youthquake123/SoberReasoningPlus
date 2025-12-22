import pandas as pd
import argparse
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_parquet_to_csv(parquet_path):
    """
    Converts a Parquet file to a CSV file in the same directory.

    Args:
        parquet_path (str): The path to the input Parquet file.
    """
    if not os.path.exists(parquet_path):
        logging.error(f"File not found: {parquet_path}")
        return False

    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_path)

        # Define the output CSV path
        directory = os.path.dirname(parquet_path)
        filename = os.path.splitext(os.path.basename(parquet_path))[0]
        csv_path = os.path.join(directory, f"{filename}.csv")

        # Convert to CSV
        df.to_csv(csv_path, index=False)
        logging.info(f"Converted: {parquet_path} -> {csv_path}")
        return True

    except Exception as e:
        logging.error(f"Error converting {parquet_path}: {e}")
        return False


def convert_directory(directory, recursive=True):
    """
    Converts all Parquet files in a directory to CSV files.

    Args:
        directory (str): The path to the directory.
        recursive (bool): If True, search subdirectories recursively.
    """
    directory = Path(directory)
    if not directory.exists():
        logging.error(f"Directory not found: {directory}")
        return

    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = list(directory.glob(pattern))

    if not parquet_files:
        logging.warning(f"No parquet files found in {directory}")
        return

    logging.info(f"Found {len(parquet_files)} parquet files")

    success_count = 0
    for parquet_path in parquet_files:
        if convert_parquet_to_csv(str(parquet_path)):
            success_count += 1

    logging.info(f"Conversion complete: {success_count}/{len(parquet_files)} files converted")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet file(s) to CSV file(s)."
    )
    parser.add_argument(
        "path", type=str, help="Path to a Parquet file or directory containing Parquet files."
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do not search subdirectories recursively (only applies to directory input)."
    )

    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        convert_parquet_to_csv(str(path))
    elif path.is_dir():
        convert_directory(path, recursive=not args.no_recursive)
    else:
        logging.error(f"Path does not exist: {args.path}")


if __name__ == "__main__":
    main()

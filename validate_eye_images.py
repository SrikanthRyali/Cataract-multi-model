import os
import csv

from app import basic_image_validation


def main():
    dataset_root = "Dataset"
    output_csv = "eye_validation_report.csv"

    # open CSV up front and write header; we'll append rows periodically
    # use utf-8 to handle emoji/error messages
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "split", "class", "is_valid", "error_message"])

        buffer = []
        count = 0
        t_cnt = 0

        # walk through the Train/Test splits and class subfolders
        for split in ("Train", "Test"):
            for cls in ("Cataract", "Normal"):
                folder = os.path.join(dataset_root, split, cls)
                if not os.path.isdir(folder):
                    continue
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    if not os.path.isfile(fpath):
                        continue
                    # run the validator
                    is_valid, error_msg = basic_image_validation(fpath)
                    t_cnt += 1
                    print("running...", fname, t_cnt, "->", is_valid)
                    buffer.append((fpath, split, cls, is_valid, error_msg))
                    count += 1

                    # flush every 10 rows
                    if count % 10 == 0:
                        writer.writerows(buffer)
                        buffer.clear()
        # write any remaining rows
        if buffer:
            writer.writerows(buffer)

    print(f"Finished validation. Report written to {output_csv}")


if __name__ == "__main__":
    main()

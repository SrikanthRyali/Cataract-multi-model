import csv
import os

# Path to the CSV file
csv_file = 'eye_validation_report.csv'

# Read the CSV and collect files to delete
files_to_delete = []
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['is_valid'].lower() == 'false':
            files_to_delete.append(row['filepath'])

# Delete the files
for filepath in files_to_delete:
    full_path = os.path.join(os.getcwd(), filepath.replace('\\', os.sep))
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"Deleted: {full_path}")
    else:
        print(f"File not found: {full_path}")

print(f"Total files deleted: {len(files_to_delete)}")
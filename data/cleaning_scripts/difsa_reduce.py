import sys
import os
from shutil import copyfile
import csv

def main():

    csv_dir = sys.argv[1]
    image_dir = sys.argv[2]
    output_dir = sys.argv[3]

    csvs = sorted(os.listdir(csv_dir))

    for subj in csvs:
        file_path = csv_dir + '/' + subj
        csv_file = open(file_path, 'r')
        reader = csv.reader(csv_file, delimiter=',')

        for row in reader:
            frame = row[0]
            if not is_row_zero(row[1:]):
                copy_frame_data(frame, row, subj, image_dir, output_dir)

        csv_file.close()


def copy_frame_data(frame, data, subj, image_dir, output_dir):
    src_image_path = "%s/%s/%s_%s"%(image_dir, subj, subj, frame)
    dest_image_path = "%s/Images/%s/%s_%s"%(output_dir, subj, subj, frame)
    new_csv_path =    "%s/AUs/%s"

    new_csv = open(new_csv_path, 'a+')
    row = ", ".join(data) + "\n"
    new_csv.write(row)

    if not os.path.exists(os.path.dirname(dest_image_path)):
        os.makedirs(os.path.dirname(dest_image_path))

    copyfile(src_image_path, dest_image_path)

    new_csv.close()

def is_row_zero(row):
    for item in row:
        if item != 0:
            return False
    return True


if __name__ == '__main__':
    main()
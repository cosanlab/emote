import csv
import sys
import os

def main():
    path = sys.argv[1]

    frames = []

    for j, file in enumerate(sorted(os.listdir(path))):
        csvfile = open(path + '/' + file)
        
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if j == 0:
                frames.append([])
                frames[i].append(i+1)

            frames[i].append(row[1])

        csvfile.close()

    out_file = open(sys.argv[2], 'w')

    writer = csv.writer(out_file, delimiter=',')
    writer.writerows(frames)

if __name__ == '__main__':
    main()
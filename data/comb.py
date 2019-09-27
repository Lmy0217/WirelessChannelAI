import csv
import os


def comb(data_path, save_file):
    with open(save_file, 'w', newline='') as sf:
        sf_csv = csv.writer(sf)
        for root, _, files in os.walk(data_path):
            for file in files:
                print(file)
                with open(os.path.join(root, file), 'r') as f:
                    f_csv = csv.reader(f)
                    for i, row in enumerate(f_csv):
                        if i > 0:
                            sf_csv.writerow(row)


if __name__ == '__main__':
    comb('./train_set', './train.csv')

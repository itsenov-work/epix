import json
import csv


if __name__ == '__main__':
    with open('resources/type_2_diabetes_selected_features.csv') as f:
        lines = f.readlines()[1:]
        lines = set([l.strip("\n") for l in lines])

    with open('resources/diabetes_cpg_list.csv') as f:
        diabetes_lines = f.readlines()[1:]

    diabetes_lines = set([l.split(',')[0] for l in diabetes_lines])

    print(diabetes_lines)
    print(lines)
    print(lines.intersection(diabetes_lines))

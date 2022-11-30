from ewas.ewas_database import save_json


def calculate_intersection_graph(sorted_cpg, diabetes_cpg):
    intersections = []

    expected_per_1000 = len(diabetes_cpg) / len(cpgs) * 1000
    print(f'Expected per 1000: {expected_per_1000}')

    for i in range(0, 385000, 1000):
        sorted_cpg_i = sorted_cpg[:i]
        intersection = set(sorted_cpg_i).intersection(diabetes_cpg)
        intersections.append(len(intersection))

    intersection_diff_from_random = [intersection / (expected_per_1000 * (i+0.01)) for i, intersection in enumerate(intersections)]
    import matplotlib.pyplot as plt
    plt.plot(intersection_diff_from_random)
    plt.show()


if __name__ == '__main__':
    with open('../resources/diabetes_fclassif.csv') as f:
        cpgs = f.readline().strip('\n').split(',')[1:]
        correlations = f.readline().strip('\n').split(',')[1:]

    print(len(cpgs))
    print(len(correlations))
    zipped = zip(cpgs, correlations)
    zipped = sorted(zipped, key=lambda x: float(x[1]), reverse=True)
    with open('../resources/diabetes_cpg_list.csv') as f:
        diabetes_lines = f.readlines()[1:]

    diabetes_lines = set([l.split(',')[0] for l in diabetes_lines])
    diabetes_lines = diabetes_lines.intersection(set(cpgs))
    print(f'Diabetes lines: {len(diabetes_lines)}')
    print(f'Diabetes lines by correlation:')
    cpg_to_correlation = dict(zipped)
    diabetes_correlations = [(cpg, cpg_to_correlation[cpg]) for cpg in diabetes_lines]
    diabetes_correlations = sorted(diabetes_correlations, key=lambda x: float(x[1]), reverse=True)
    print(len(diabetes_correlations))
    print(diabetes_correlations)
    json_correlations = []
    for x in diabetes_correlations:
        json_correlations.append({
            'cpg': x[0],
            'correlation_to_diabetes': float(x[1])
        })
    save_json('statistics', 'momchil_diabetes_correlations', json_correlations)
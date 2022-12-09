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
    plt.savefig(f'../resources/plot_{filename_diabetes}_{filename_classif}.jpg')
    plt.show()
    plt.close()


def read_diabetes_cpgs(filename):
    with open(f'../resources/{filename}.tsv', 'r') as f:
        cpgs = [line.split('\t')[1] for line in f.readlines()[1:]]
    return cpgs


if __name__ == '__main__':

    filename_diabetes = 'COMMON_CPG_OVERLAP_DIABETES'
    filename_classif = 'diabetes_chi2'
    with open(f'../resources/{filename_classif}.csv') as f:
        cpgs = f.readline().strip('\n').split(',')[1:]
        correlations = f.readline().strip('\n').split(',')[1:]

    print(len(cpgs))
    print(len(correlations))
    zipped = zip(cpgs, correlations)
    zipped = sorted(zipped, key=lambda x: float(x[1]), reverse=True)

    sorted_cpgs = [c[0] for c in zipped]

    diabetes_cpgs = read_diabetes_cpgs(filename_diabetes)
    diabetes_cpgs = list(
        set(diabetes_cpgs).intersection(set(cpgs))
    )

    cpg_to_correlation = dict(zipped)
    diabetes_correlations = [(cpg, cpg_to_correlation[cpg]) for cpg in diabetes_cpgs]
    diabetes_correlations = sorted(diabetes_correlations, key=lambda x: float(x[1]), reverse=True)

    calculate_intersection_graph(sorted_cpgs, diabetes_cpgs)

    print(len(diabetes_correlations))
    print(diabetes_correlations)
    json_correlations = []
    for x in diabetes_correlations:
        json_correlations.append({
            'cpg': x[0],
            'correlation_to_diabetes': float(x[1])
        })
    save_json('statistics', f'stats_{filename_diabetes}_{filename_classif}', json_correlations)

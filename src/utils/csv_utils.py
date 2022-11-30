
def read_csv_line(l):
    return l.strip('\n').split(',')


def read_csv_file(f):
    return read_csv_line(f.readline())
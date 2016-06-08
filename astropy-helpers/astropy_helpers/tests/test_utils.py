import os
from ..utils import find_data_files


def test_find_data_files(tmpdir):

    data = tmpdir.mkdir('data')
    sub1 = data.mkdir('sub1')
    sub2 = data.mkdir('sub2')
    sub3 = sub1.mkdir('sub3')

    for directory in (data, sub1, sub2, sub3):
        filename = directory.join('data.dat').strpath
        with open(filename, 'w') as f:
            f.write('test')

    filenames = find_data_files(data.strpath, '**/*.dat')

    filenames = sorted(os.path.relpath(x, data.strpath) for x in filenames)

    assert filenames[0] == os.path.join('data.dat')
    assert filenames[1] == os.path.join('sub1', 'data.dat')
    assert filenames[2] == os.path.join('sub1', 'sub3', 'data.dat')
    assert filenames[3] == os.path.join('sub2', 'data.dat')

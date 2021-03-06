#!/usr/bin/env python
"""
python make_mlp_pkl.py <target number>

This script looks for three files like:

    skim_data_learn_target0.dat
    skim_data_test_target0.dat
    skim_data_valid_target0.dat

where "0" is the "target number" provided as an argument.
"""
from __future__ import print_function
import cPickle
import numpy as np
import sys
import os
import gzip

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if not len(sys.argv) == 2:
    print('The target number argument is mandatory.')
    print(__doc__)
    sys.exit(1)

targnum = sys.argv[1]


def get_data_from_file(filename):
    targs = []
    data = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            rowdat = []
            targs.append(int(elems[0]))
            rowdat = np.asarray(elems[1:], dtype=np.float32)
            data.append(rowdat)

    targs = np.asarray(targs)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs)  # no need to zip, just store as a tuple
    return storedat


fileroots = ['skim_data_learn_target',
             'skim_data_test_target',
             'skim_data_valid_target']
final_data = []

for filer in fileroots:
    filen = filer + targnum + '.dat'
    dtuple = get_data_from_file(filen)
    final_data.append(dtuple)

filepkl = 'skim_data_target' + targnum + '.pkl.gz'

if os.path.exists(filepkl):
    os.remove(filepkl)

storef = gzip.open(filepkl, 'wb')
cPickle.dump(final_data, storef, protocol=cPickle.HIGHEST_PROTOCOL)
storef.close()

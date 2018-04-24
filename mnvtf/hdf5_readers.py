import h5py
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


class MnvHDF5Reader:
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.

    Note that this class assumes a file structure like:
    f[group_1][dataset_1]
    f[group_1][dataset_2]
    ...
    f[group_n][dataset_m]
    """
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None
        self._dd = None
        self._groups = None

    def open(self):
        LOGGER.info("Opening hdf5 file {}".format(self.file))
        self._f = h5py.File(self.file, 'r')
        for group in self._f:
            for dset in self._f[group]:
                LOGGER.info('{:>12}/{:>12}: {}: shape = {}'.format(
                    group, dset,
                    np.dtype(self._f[group][dset]),
                    np.shape(self._f[group][dset])
                ))
        self._groups = [group for group in self._f]
        self._dd = {}
        for g in self._groups:
            self._dd[g] = [dset for dset in self._f[g]]

    def close(self):
        try:
            self._f.close()
            self._dd = None
            self._groups = None
        except AttributeError:
            LOGGER.info('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        try:
            for g in self._groups:
                if name in self._dd[g]:
                    group = g
                    break
            else:
                raise ValueError(
                    '{} is not present in any group.'.format(name)
                )
            return self._f[group][name][start_idx: stop_idx]
        except KeyError:
            LOGGER.info('{} data is not available in this HDF5 file.')
            return []

    def get_nevents(self, group):
        sizes = [self._f[group][d].shape[0] for d in self._f[group]]
        if min(sizes) != max(sizes):
            msg = "All dsets must have the same size!"
            LOGGER.error(msg)
            raise ValueError(msg)
        return sizes[0]


def test():
    path = '/Users/perdue/Documents/MINERvA/AI/hdf5/201801/'
    filen = 'hadmultkineimgs_127x94_me1Amc_tiny.hdf5'
    reader = MnvHDF5Reader(path + filen)
    try:
        reader.open()
        print(reader.get_data('eventids', 0, 10))
        try:
            print(reader.get_data('n_taus', 0, 10))
        except ValueError:
            print('n_taus is not stored here.')
        try:
            print(reader.get_data('n_rainbow_unicorns', 0, 10))
        except ValueError:
            print('of course there are no unicorns here.')
        reader.close()
    except IOError:
        print('hdf5 file not found')


if __name__ == '__main__':
    test()

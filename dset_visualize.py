#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py -f [file name] -n [optional: # of evts, def==10]
"""
import sys
from collections import OrderedDict
import pylab
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
import numpy as np

from mnvtf.utils import get_reader_class
from mnvtf.utils import make_data_reader_dict
from mnvtf.data_constants import HITIMESU, HITIMESV, HITIMESX
from mnvtf.data_constants import PIDU, PIDV, PIDX
from mnvtf.data_constants import EVENT_DATA, EVENTIDS
from mnvtf.data_constants import PLANECODES, SEGMENTS, ZS
from mnvtf.data_constants import N_HADMULTMEAS
from mnvtf.data_constants import SEGMENTATION_TYPE
from mnvtf.hdf5_readers import MnvHDF5Reader

from evtid_utils import decode_eventid


class MnvDataReader:
    def __init__(
            self,
            filename,
            n_events=10,
            views=['x', 'u', 'v'],
            img_sizes=(94, 47),
            n_planecodes=173,
            tfrecord_reader_type=None,
            data_format='NHWC',
            seg_data=False
    ):
        """
        currently, only work with compressed tfrecord files; assume compression
        for hdf5 is inside, etc.
        """
        self._f = None
        self.filename = filename
        self.n_events = n_events
        self.views = views
        self.img_sizes = img_sizes
        self.n_planecodes = n_planecodes
        self.img_shp = (127, img_sizes[0], img_sizes[1], 2)
        self.data_format = data_format
        self.seg_data = seg_data

        ext = self.filename.split('.')[-1]
        self.compression = ext if ext in ['zz', 'gz'] else ''
        if self.compression in ['zz', 'gz']:
            self.filetype = filename.split('.')[-2]
        else:
            self.filetype = ext

        self.hdf5_extensions = ['hdf5', 'h5']
        self.tfr_extensions = ['tfrecord']

        if tfrecord_reader_type is None:
            # attempt to infer the reader type from the filename.
            tfrecord_reader_type = filename.split('/')[-1]
            tfrecord_reader_type = tfrecord_reader_type.split('_')[0]
        self.tfrecord_reader = get_reader_class(tfrecord_reader_type)

    def _read_tfr(self):
        data_dict = {}
        data_dict['energies+times'] = {}

        dd = make_data_reader_dict(
            filenames_list=[self.filename],
            batch_size=self.n_events,
            name='test_read',
            compression=self.compression,
            img_shp=self.img_shp,
            data_format=self.data_format,
            n_planecodes=self.n_planecodes
        )
        reader = self.tfrecord_reader(dd)
        # get an ordered dict
        batch_dict = reader.batch_generator()

        def tp_tnsr(tnsr):
            return np.transpose(tnsr, [0, 3, 1, 2])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                tensor_list = sess.run(batch_dict.values())
                results = OrderedDict(zip(batch_dict.keys(), tensor_list))
                data_dict['energies+times']['x'] = tp_tnsr(results[HITIMESX])
                data_dict['energies+times']['u'] = tp_tnsr(results[HITIMESU])
                data_dict['energies+times']['v'] = tp_tnsr(results[HITIMESV])
                data_dict[EVENTIDS] = results[EVENTIDS]
                if ZS in results.keys():
                    data_dict[ZS] = results[ZS]
                # need to 'de-one-hot' these...
                for k in [PLANECODES, SEGMENTS, N_HADMULTMEAS]:
                    if k in results.keys():
                        data_dict[k] = np.argmax(
                            results[k], axis=1
                        ).reshape(results[k].shape[0], 1)
            except tf.errors.OutOfRangeError:
                print('Reading stopped - queue is empty.')
            finally:
                coord.request_stop()
                coord.join(threads)

        return data_dict

    def _read_hdf5(self):
        """
        possibilities: energy tensors, time tensors, energy+time tensors
        (2-deep). get everything there into a dictionary keyed by type,
        and then by view.
        """
        data_dict = {}
        data_dict['energies+times'] = {}

        m = MnvHDF5Reader(self.filename)
        m.open()
        n_events = m.get_nevents(group=EVENT_DATA)
        n_read = min(n_events, self.n_events)
        data_dict['energies+times']['x'] = m.get_data(HITIMESX, 0, n_read)
        data_dict['energies+times']['u'] = m.get_data(HITIMESU, 0, n_read)
        data_dict['energies+times']['v'] = m.get_data(HITIMESV, 0, n_read)
        data_dict[EVENTIDS] = m.get_data(EVENTIDS, 0, n_read)
        
        if self.seg_data:
            data_dict['pid'] = {}
            data_dict['pid']['x'] = m.get_data(PIDX, 0, n_read)
            data_dict['pid']['u'] = m.get_data(PIDU, 0, n_read)
            data_dict['pid']['v'] = m.get_data(PIDV, 0, n_read)

        def get_hdf_dat(hdf_key):
            v = m.get_data(hdf_key, 0, n_read)
            return v if len(v) else None
        for d in [PLANECODES, SEGMENTS, ZS, N_HADMULTMEAS]:
            v = get_hdf_dat(d)
            if v is not None and len(v):
                data_dict[d] = v

        m.close()

        return data_dict

    def read_data(self):
        """
        return a dictionary of ndarrays, keyed by 'x', 'u', and 'v',
        each with shape (N, C, H, W) - could be anywhere from 1 to 3 views.
        """
        if self.filetype in self.hdf5_extensions:
            return self._read_hdf5()
        elif self.filetype in self.tfr_extensions:
            return self._read_tfr()
        else:
            raise ValueError('Invalid file type extension!')


def make_plots(data_dict, max_events, normed_img, pred_dict):
    """
    cases:
    * 'energies+times',
    * 'energies' and 'times' separately,
    * or 'energies' or 'times'
    If 2-deep tensor, assume energy is index 0, time is index 1
    """
    target_plane_codes = {9: 1, 18: 2, 27: 3, 36: 6, 45: 4, 50: 5}
    pkeys = []
    for k in data_dict.keys():
        if len(data_dict[k]) > 0:
            pkeys.append(k)
    print('Data dictionary present keys: {}'.format(pkeys))

    types = ['energy', 'time']
    views = ['x', 'u', 'v']   # TODO? build dynamically?

    # only working with two-deep imgs these days
    # plotting_two_tensors = True

    def get_maybe_missing(data_dict, key, counter):
        try:
            return data_dict[key][counter]
        except KeyError:
            pass
        return -1

    evt_plotted = 0
    for counter in range(len(data_dict[EVENTIDS])):
        evtid = data_dict[EVENTIDS][counter]
        segment = get_maybe_missing(data_dict, SEGMENTS, counter)
        planecode = get_maybe_missing(data_dict, PLANECODES, counter)
        n_hadmultmeas = get_maybe_missing(data_dict, N_HADMULTMEAS, counter)
        (run, subrun, gate, phys_evt) = decode_eventid(evtid)
        if evt_plotted > max_events:
            break
        status_string = 'Plotting entry %d: %d: ' % (counter, evtid)
        title_string = '{}/{}/{}/{}'
        title_elems = [run, subrun, gate, phys_evt]
        if segment != -1 and planecode != -1:
            title_string = title_string + ', segment {}, planecode {}'
            title_elems.extend([segment, planecode])
            if planecode in tuple(target_plane_codes.keys()):
                title_string = title_string + ', targ {}'
                title_elems.append(target_plane_codes[planecode[0]])
        if n_hadmultmeas != -1:
            title_string = title_string + ', n_chghad {}'
            title_elems.append(n_hadmultmeas)
        if pred_dict is not None:
            try:
                prediction = pred_dict[str(evtid)]
                title_string = title_string + ', pred={}'
                title_elems.append(prediction)
            except KeyError:
                pass
        print(status_string + title_string.format(*title_elems))

        # run, subrun, gate, phys_evt = decode_eventid(evtid)
        fig_wid = 9
        fig_height = 6
        grid_height = 2
        fig = pylab.figure(figsize=(fig_wid, fig_height))
        fig.suptitle(title_string.format(*title_elems))
        gs = pylab.GridSpec(grid_height, 3)

        for i, t in enumerate(types):
            datatyp = 'energies+times'
            # set the bounds on the color scale
            if normed_img:
                minv = 0 if t == 'energy' else -1
                maxv = 1
            else:
                maxes = []
                mins = []
                for v in views:
                    maxes.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                    mins.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                minv = np.max(mins)
                maxv = np.max(maxes)
                maxex = maxv if maxv > minv else minv
                minv = 0 if minv < 0.0001 else 0 if t == 'energy' else -maxv
                maxv = maxex
            for j, view in enumerate(views):
                gs_pos = i * 3 + j
                ax = pylab.subplot(gs[gs_pos])
                ax.axis('on')
                ax.xaxis.set_major_locator(pylab.NullLocator())
                ax.yaxis.set_major_locator(pylab.NullLocator())
                cmap = 'Reds' if t == 'energy' else 'bwr'
                cbt = 'energy' if t == 'energy' else 'times'
                datap = data_dict[datatyp][view][counter, i, :, :]
                # make the plot
                im = ax.imshow(
                    datap,
                    cmap=pylab.get_cmap(cmap),
                    interpolation='nearest',
                    vmin=minv, vmax=maxv
                )
                cbar = pylab.colorbar(im, fraction=0.04)
                cbar.set_label(cbt, size=9)
                cbar.ax.tick_params(labelsize=6)
                pylab.title(t + ' - ' + view, fontsize=12)
                pylab.xlabel('plane', fontsize=10)
                pylab.ylabel('strip', fontsize=10)
        figname = 'evt_%d.pdf' % (counter)
        pylab.savefig(figname, bbox_inches='tight')
        pylab.close()
        evt_plotted += 1


def make_plots_seg(data_dict, max_events, normed_img, pred_dict):
    """
    Copy of make_plots adapted for pid plots
    """
    target_plane_codes = {9: 1, 18: 2, 27: 3, 36: 6, 45: 4, 50: 5}
    pkeys = []
    for k in data_dict.keys():
        if len(data_dict[k]) > 0:
            pkeys.append(k)
    print('Data dictionary present keys: {}'.format(pkeys))

    types = ['energy', 'time']
    views = ['x', 'u', 'v']   # TODO? build dynamically?

    # only working with two-deep imgs these days
    # plotting_two_tensors = True

    def get_maybe_missing(data_dict, key, counter):
        try:
            return data_dict[key][counter]
        except KeyError:
            pass
        return -1

    evt_plotted = 0

    with PdfPages("evt_all.pdf") as pdf:
        for counter in range(len(data_dict[EVENTIDS])):
            evtid = data_dict[EVENTIDS][counter]
            segment = get_maybe_missing(data_dict, SEGMENTS, counter)
            planecode = get_maybe_missing(data_dict, PLANECODES, counter)
            n_hadmultmeas = get_maybe_missing(data_dict, N_HADMULTMEAS, counter)
            (run, subrun, gate, phys_evt) = decode_eventid(evtid)
            if evt_plotted > max_events:
                break
            status_string = 'Plotting entry %d: %d: ' % (counter, evtid)
            title_string = '{}/{}/{}/{}'
            title_elems = [run, subrun, gate, phys_evt]
            if segment != -1 and planecode != -1:
                title_string = title_string + ', segment {}, planecode {}'
                title_elems.extend([segment, planecode])
                if planecode in tuple(target_plane_codes.keys()):
                    title_string = title_string + ', targ {}'
                    title_elems.append(target_plane_codes[planecode[0]])
            if n_hadmultmeas != -1:
                title_string = title_string + ', n_chghad {}'
                title_elems.append(n_hadmultmeas)
            if pred_dict is not None:
                try:
                    prediction = pred_dict[str(evtid)]
                    title_string = title_string + ', pred={}'
                    title_elems.append(prediction)
                except KeyError:
                    pass
            print(status_string + title_string.format(*title_elems))

            # run, subrun, gate, phys_evt = decode_eventid(evtid)
            fig_wid = 9
            fig_height = 9
            grid_height = 3
            fig = pylab.figure(figsize=(fig_wid, fig_height))
            fig.suptitle(title_string.format(*title_elems))
            gs = pylab.GridSpec(grid_height, 3)

            for i, t in enumerate(types):
                datatyp = 'energies+times'
                # set the bounds on the color scale
                if normed_img:
                    minv = 0 if t == 'energy' else -1
                    maxv = 1
                else:
                    maxes = []
                    mins = []
                    for v in views:
                        maxes.append(
                            np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                        )
                        mins.append(
                            np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                        )
                    minv = np.max(mins)
                    maxv = np.max(maxes)
                    maxex = maxv if maxv > minv else minv
                    minv = 0 if minv < 0.0001 else 0 if t == 'energy' else -maxv
                    maxv = maxex
                for j, view in enumerate(views):
                    gs_pos = i * 3 + j
                    ax = pylab.subplot(gs[gs_pos])
                    ax.axis('on')
                    ax.xaxis.set_major_locator(pylab.NullLocator())
                    ax.yaxis.set_major_locator(pylab.NullLocator())
                    cmap = 'Reds' if t == 'energy' else 'bwr'
                    cbt = 'energy' if t == 'energy' else 'times'
                    datap = data_dict[datatyp][view][counter, i, :, :]
                    # make the plot
                    im = ax.imshow(
                        datap,
                        cmap=pylab.get_cmap(cmap),
                        interpolation='nearest',
                        vmin=minv, vmax=maxv
                    )
                    cbar = pylab.colorbar(im, fraction=0.04)
                    cbar.set_label(cbt, size=9)
                    cbar.ax.tick_params(labelsize=6)
                    pylab.title(t + ' - ' + view, fontsize=12)
                    pylab.xlabel('plane', fontsize=10)
                    pylab.ylabel('strip', fontsize=10)
            # plot pid
            for j, view in enumerate(views):
                gs_pos = 6 + j
                ax = pylab.subplot(gs[gs_pos])
                ax.axis('on')
                ax.xaxis.set_major_locator(pylab.NullLocator())
                ax.yaxis.set_major_locator(pylab.NullLocator())
                cmap = 'tab10'
                cbt = 'pid'
                datap = data_dict["pid"][view][counter, 0, :, :]
                # make the plot                
                im = ax.imshow(
                    datap,
                    cmap=pylab.get_cmap(cmap),
                    interpolation='nearest',
                    vmin=0, vmax=7
                )
                cbar = pylab.colorbar(im, fraction=0.04, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
                cbar.ax.set_yticklabels(['nth', 
                                         'EM',
                                         'mu',
                                         'pi+',
                                         'pi-',
                                         'n',
                                         'p',
                                         'oth'])
                cbar.set_label("pid", size=9)
                cbar.ax.tick_params(labelsize=6)
                pylab.title("pid" + ' - ' + view, fontsize=12)
                pylab.xlabel('plane', fontsize=10)
                pylab.ylabel('strip', fontsize=10)
            
            pdf.savefig()
            evt_plotted += 1


def get_predictions(pred_filename, n_items=200):
    pd = {}
    with open(pred_filename, 'r') as f:
        for _ in range(n_items):
            l = f.readline()
            its = l.split(',')
            evtid = its[0] + its[1] + its[2] + its[3]
            pred = its[4]
            pd[evtid] = pred
    return pd


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-f', '--file', dest='filename',
                      help='Dset file name', metavar='FILENAME',
                      default=None, type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=10,
                      help='Number of events', metavar='N_EVENTS',
                      type='int')
    parser.add_option('--imgw_x', dest='imgw_x', default=94,
                      help='Image width (x)', metavar='IMG_WIDTHX',
                      type='int')
    parser.add_option('--imgw_uv', dest='imgw_uv', default=47,
                      help='Image width (uv)', metavar='IMG_WIDTHUV',
                      type='int')
    parser.add_option('--n_planecodes', dest='n_planecodes', default=173,
                      help='Number of planecodes (onehot)',
                      metavar='N_PLANECODES', type='int')
    parser.add_option('--normed_img', dest='normed_img', default=False,
                      help='Image from normalized source',
                      metavar='NORMED_IMG', action='store_true')
    parser.add_option('-p', '--predictions', dest='predictions_file',
                      help='Predictions file name', metavar='PREDICTIONS',
                      default=None, type='string')
    parser.add_option('-t', '--reader_type', dest='reader_type',
                      help='Reader type (see mnvtf.utils.get_reader_class for available options', metavar='READER',
                      default=None, type='string')

    (options, args) = parser.parse_args()

    if not options.filename:
        print("\nSpecify file (-f):\n\n")
        print(__doc__)
        sys.exit(1)

    img_sizes = (options.imgw_x, options.imgw_uv)
    reader = MnvDataReader(
        filename=options.filename,
        n_events=options.n_events,
        img_sizes=img_sizes,
        n_planecodes=options.n_planecodes,
        tfrecord_reader_type=options.reader_type,
        seg_data=(options.reader_type == SEGMENTATION_TYPE)
    )
    dd = reader.read_data()

    if options.predictions_file:
        pd = get_predictions(options.predictions_file, options.n_events)
    else:
        pd = None

    if options.reader_type == SEGMENTATION_TYPE:
        make_plots_seg(dd, options.n_events, options.normed_img, pd)
    else:
        make_plots(dd, options.n_events, options.normed_img, pd)

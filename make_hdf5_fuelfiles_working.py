#!/usr/bin/env python
"""

Execution:
    python make_hdf5_fuelfiles.py -b 'base name' -o 'output'

The default output name is 'minerva_fuel.hdf5'. The default skim type is
'nukecc_vtx'. Default image width and heights are 127 x 94. Target padding
is not included by default.

CAUTION: If you include target padding, you must be sure to adjust the
`--trim_column_up_x` (default 0) and `--trim_column_down_x` (default 94)
values appropriately.

Notes:
------
* imgw, imgh - "pixel" size of data images: here - H corresponds to MINERvA Z,
and W correpsonds to the view axis

* The 'nukecc_vtx' `--skim` option expects data layout like:
    # 0   1   2   3   4   5     6     7
    # seg z   pln run sub gate  slice data (X:[E] U:[E] ... etc.)...

"""
from __future__ import print_function
import sys
import os
import re
from collections import OrderedDict
import gzip

import numpy as np
import h5py

from fuel.datasets.hdf5 import H5PYDataset

from six.moves import range

from plane_codes import build_indexed_codes

PDG_PROTON = 2212
PDG_PIPLUS = 211
PDG_PIMINUS = -211
PDG_KPLUS = 321
PDG_KMINUS = -321
PDG_SIGMAPLUS = 3222
PDG_SIGMAMINIUS = 3112

PDG_ELECTRON = 11
PDG_NUE = 12
PDG_MUON = 13
PDG_NUMU = 14
PDG_TAU = 15
PDG_NUTAU = 16

PDG_POSITRON = -11
PDG_NUEBAR = -12
PDG_ANTIMUON = -13
PDG_NUMUBAR = -14
PDG_ANTITAU = -15
PDG_NUTAUBAR = -16


def compute_target_padding():
    """
    When adding padding, we traverse 8 planes before reaching target1,
    then 8 planes before reaching target 2, then 8 planes before reaching
    target 3, which we count as 2 planes thick, then 8 planes before
    reaching the water target, which we regard as 6 modules thick, then
    8 planes before reaching target 4, then 4 planes before reaching
    target 5.

    return tuple of lists - first list is the locations in minerva plane
    occurence index (start with zero, count up as we go along) where we
    should skip two planes, and the second is the list of locations
    where we should skip four

    note in the steps below we are always traveling in steps of groups
    of four - uxvx - so we will only count _x_ planes we have _traversed_
    while building these steps so when we loop and insert padding we have
    moved the right amount of space through the detector (because u and v
    are encoded "in step" with x, with one layer of padding already inserted
    between each u or v to account for the sparsity in those views.
    """
    base_steps = 1      # insert _after_ traversing n cols, etc.
    target1_steps = (8 // 2) + base_steps
    target2_steps = (8 // 2) + target1_steps
    target3_steps = (8 // 2) + target2_steps
    water_steps = (8 // 2) + target3_steps
    target4_steps = (8 // 2) + water_steps
    target5_steps = (4 // 2) + target4_steps
    two_breaks = [target1_steps, target2_steps,
                  target4_steps, target5_steps]
    four_breaks = [target3_steps]
    six_breaks = [water_steps]
    return two_breaks, four_breaks, six_breaks


def get_total_target_padding():
    """
    get the sum of all the spaces following the "breaks" where targets sit
    """
    two_breaks, four_breaks, six_breaks = compute_target_padding()
    target_padding = 2 * len(two_breaks) + 4 * len(four_breaks) + \
        6 * len(six_breaks)
    return target_padding


def get_output_imgh(imgh, add_target_padding=False):
    if add_target_padding:
        imgh += get_total_target_padding()
    return imgh


def pad_for_targets(imgw, imgh, hitsX, hitsU, hitsV):
    two_breaks, four_breaks, six_breaks = compute_target_padding()
    imgh_padding = get_total_target_padding()
    tempX = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)
    tempU = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)
    tempV = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)

    def col_copy(frm, to):
        tempX[:, to] = hitsX[:, frm]
        tempU[:, to] = hitsU[:, frm]
        tempV[:, to] = hitsV[:, frm]

    shifted_column_counter = 0
    for i in range(imgh):
        j = i + 1
        if j in two_breaks:
            shifted_column_counter += 2
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        elif j in four_breaks:
            shifted_column_counter += 4
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        elif j in six_breaks:
            shifted_column_counter += 6
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        else:
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
    return tempX, tempU, tempV


def trim_columns(img, col_up, col_dn):
    """
    keep columns between `col_up` and `col_dn`
    """
    img = img[:, col_up:col_dn]
    return img


def shape_and_flip_image(hits, imgw, imgh):
    hits = np.asarray(hits, dtype=np.float32).reshape(imgw, imgh)
    # we're "upside down" by default in images, flip back just
    # so things look normal - shouldn't affect anything
    hits = hits[::-1, :]
    return hits


def shift_img_updown(img, shift):
    """
    input images are expected to be three-tensors like `(1, imgw, imgh)`
    """
    base_w = np.shape(img)[1]
    new_image = np.zeros_like(img)
    from_row1 = 0
    from_row2 = base_w
    to_row1 = 0
    to_row2 = base_w
    if shift > 0:
        from_row1 += shift
        to_row2 = base_w - shift
    else:
        from_row2 = base_w + shift
        to_row1 += -shift

    new_image[:, to_row1:to_row2, :] = img[:, from_row1:from_row2, :]
    return new_image


def unpack_xuv_skim_data(rowdat, imgw, imgh, add_target_padding,
                         trims, insert_x_padding_into_uv):
    hitsX = []
    hitsU = []
    hitsV = []
    imgh_x = 0
    imgh_u = 0
    imgh_v = 0
    for point in rowdat:
        hit_data = point.split(':')
        view = hit_data[0]
        energy = float(hit_data[1])
        if view == 'X':
            hitsX.append(energy)
            imgh_x += 1
        elif view == 'U':
            hitsU.append(energy)
            imgh_u += 1
            if insert_x_padding_into_uv:
                hitsU.append(0.0)
                imgh_u += 1
        elif view == 'V':
            hitsV.append(energy)
            imgh_v += 1
            if insert_x_padding_into_uv:
                hitsV.append(0.0)
                imgh_v += 1

    imgh_x = imgh_x // imgw
    imgh_u = imgh_u // imgw
    imgh_v = imgh_v // imgw
    hitsX = shape_and_flip_image(hitsX, imgw, imgh_x)
    hitsU = shape_and_flip_image(hitsU, imgw, imgh_u)
    hitsV = shape_and_flip_image(hitsV, imgw, imgh_v)
    if add_target_padding:
        hitsX, hitsU, hitsV = pad_for_targets(imgw, imgh,
                                              hitsX, hitsU, hitsV)
    hitsX = trim_columns(hitsX, trims[0][0], trims[0][1])
    hitsU = trim_columns(hitsU, trims[1][0], trims[1][1])
    hitsV = trim_columns(hitsV, trims[2][0], trims[2][1])
    return hitsX, hitsU, hitsV


def process_particles_for_hadron_multiplicty(pdgs, energies, thresh=50):
    """
    thresh is in MeV
    """
    pdglist = [PDG_PROTON, PDG_PIPLUS, PDG_PIMINUS, PDG_KPLUS, PDG_KMINUS]
    data = []
    for i, en in enumerate(energies):
        if en > thresh and pdgs[i] in pdglist:
            dat = (pdgs[i], en)
            data.append(dat)
    return data


def get_hadmult_study_data_from_file(filename):
    print("...loading data")
    eventids = []
    n_protons_arr = []
    esum_protons_arr = []
    n_pions_arr = []
    esum_pions_arr = []
    n_kaons_arr = []
    esum_kaons_arr = []
    # format:
    # 0   1   2   3   4   5   6   7
    # run sub gt  slc data... (p:pdg:E)

    with gzip.open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            eventid = elems[0] + elems[1].zfill(4) + elems[2].zfill(4) \
                + elems[3].zfill(2)
            eventids.append(eventid)
            partdat = elems[4:]
            pdgs = []
            energies = []
            for particle in partdat:
                dat = particle.split(':')
                pdgs.append(int(dat[1]))
                energies.append(float(dat[2]))
            processed_parts = \
                process_particles_for_hadron_multiplicty(pdgs, energies)
            n_protons = 0
            n_pions = 0
            n_kaons = 0
            esum_protons = 0
            esum_pions = 0
            esum_kaons = 0
            for particle in processed_parts:
                if particle[0] == PDG_PROTON:
                    n_protons += 1
                    esum_protons += particle[1]
                elif particle[0] == PDG_PIPLUS or particle[0] == PDG_PIMINUS:
                    n_pions += 1
                    esum_pions += particle[1]
                elif particle[0] == PDG_KPLUS or particle[0] == PDG_KMINUS:
                    n_kaons += 1
                    esum_kaons += particle[1]
            n_protons_arr.append(n_protons)
            esum_protons_arr.append(esum_protons)
            n_pions_arr.append(n_pions)
            esum_pions_arr.append(esum_pions)
            n_kaons_arr.append(n_kaons)
            esum_kaons_arr.append(esum_kaons)
    eventids = np.asarray(eventids, dtype=np.uint64)
    n_protons_arr = np.asarray(n_protons_arr, dtype=np.uint8)
    n_pions_arr = np.asarray(n_pions_arr, dtype=np.uint8)
    n_kaons_arr = np.asarray(n_kaons_arr, dtype=np.uint8)
    esum_protons_arr = np.asarray(esum_protons_arr, dtype=np.float32)
    esum_protons_arr = np.asarray(esum_protons_arr, dtype=np.float32)
    esum_protons_arr = np.asarray(esum_protons_arr, dtype=np.float32)
    # pdgs = np.asarray(pdgs, dtype=np.int64)
    # energies = np.asarray(energies, dtype=np.float32)
    storedat = (n_protons_arr, esum_protons_arr,
                n_pions_arr, esum_pions_arr,
                n_kaons_arr, esum_kaons_arr,
                eventids)
    print("...finished loading")
    return storedat


def get_nukecc_vtx_study_data_from_file(filename, imgw, imgh, trims,
                                        add_target_padding=False,
                                        insert_x_padding_into_uv=True):
    """
    imgw, imgh - specify the size of the raw data image in the file

    trim_column_up - specify if we want to trim the target region of
    the detector off for tracker analysis
    trim_column_dn - specify if we want to cut the downstream part of
    the detector off to speed up target analysis

    NOTE: trim_column_up and trim_column_dn use AFTER PADDING NUMBERS!

    add_target_padding - add in blanks for the targets in the target region?

    Return (dataX, dataU, dataV, targs, zs, planeids, eventids) in that order.
    """
    print("...loading data")
    targs = []
    zs = []
    planeids = []
    eventids = []
    dataX = []
    dataU = []
    dataV = []
    icodes = build_indexed_codes()
    # format:
    # 0   1   2   3   4   5   6   7
    # seg z   pln run sub gt  slc data...

    with gzip.open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            targs.append(int(elems[0]))
            zs.append(float(elems[1]))
            rawid = int(elems[2])
            planeid = icodes[rawid]
            planeids.append(planeid)
            eventid = elems[3] + elems[4].zfill(4) + elems[5].zfill(4) \
                + elems[6].zfill(2)
            eventids.append(eventid)
            rowdat = elems[7:]
            hitsX, hitsU, hitsV = unpack_xuv_skim_data(
                rowdat, imgw, imgh, add_target_padding,
                trims, insert_x_padding_into_uv)
            dataX.append(hitsX)
            dataU.append(hitsU)
            dataV.append(hitsV)

    targs = np.asarray(targs, dtype=np.uint8)
    zs = np.asarray(zs, dtype=np.float32)
    planeids = np.asarray(planeids, dtype=np.uint16)
    eventids = np.asarray(eventids, dtype=np.uint64)
    dataX = transform_to_4d_tensor(np.asarray(dataX, dtype=np.float32))
    dataU = transform_to_4d_tensor(np.asarray(dataU, dtype=np.float32))
    dataV = transform_to_4d_tensor(np.asarray(dataV, dtype=np.float32))
    storedat = (dataX, dataU, dataV, targs, zs, planeids, eventids)
    print("...finished loading")
    return storedat


def transform_to_4d_tensor(tensr):
    shpvar = np.shape(tensr)
    shpvar = (shpvar[0], 1, shpvar[1], shpvar[2])
    tensr = np.reshape(tensr, shpvar)
    return tensr


def make_file_list(filebase, use_gzipped_files=True):
    # look for "filebase"+(_learn/_valid/_test/ - zero or more times)+whatever
    filestr = r"^%s(_learn|_test|_valid)*.*dat$"
    if use_gzipped_files:
        filestr = r"^%s(_learn|_test|_valid)*.*dat.gz$"
    filebase = re.compile(filestr % filebase)
    files = os.listdir('.')
    files = [f for f in files if re.match(filebase, f)]
    print(files)
    return files


def prepare_hdf5_file(hdf5file):
    if os.path.exists(hdf5file):
        os.remove(hdf5file)
    f = h5py.File(hdf5file, 'w')
    return f


def create_view_dset(hdf5file, name, imgw, imgh):
    data_set = hdf5file.create_dataset(name, (0, 1, imgw, imgh),
                                       dtype='float32', compression='gzip',
                                       maxshape=(None, 1, imgw, imgh))
    # `Fuel.H5PYDataset` allows us to label axes with semantic information;
    # we record that in the file using "dimensional scales" (see h5py docs)
    data_set.dims[0].label = 'batch'
    data_set.dims[1].label = 'view(xuv)'
    data_set.dims[2].label = 'height(view-coord)'
    data_set.dims[3].label = 'width(z)'


def create_1d_dset(hdf5file, name, dtype, label):
    data_set = hdf5file.create_dataset(name, (0,),
                                       dtype=dtype, compression='gzip',
                                       maxshape=(None,))
    data_set.dims[0].label = label


def add_split_dict(hdf5file, names, total_examples,
                   train_frac=0.83, valid_frac=0.10):
    # TODO: investiage the "reference" stuff so we can pluck validation
    # and testing events evenly from the sample
    final_train_index = int(total_examples * train_frac)
    final_valid_index = int(total_examples * (train_frac + valid_frac))

    train_dict = {name: (0, final_train_index)
                  for name in names}
    valid_dict = {name: (final_train_index, final_valid_index)
                  for name in names}
    test_dict = {name: (final_valid_index, total_examples)
                 for name in names}
    split_dict = {
        'train': train_dict,
        'valid': valid_dict,
        'test': test_dict
    }
    hdf5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)


def add_data_to_hdf5file(f, dset_names, dset_vals):
        examples_in_file = len(dset_vals[-1])
        print(" examples_in_file =", examples_in_file)
        existing_examples = np.shape(f[dset_names[-1]])[0]
        print(" existing_examples =", existing_examples)
        total_examples = examples_in_file + existing_examples
        print(" resize =", total_examples)
        print(" idx slice = %d:%d" % (existing_examples, total_examples))
        for name in dset_names:
            f[name].resize(total_examples, axis=0)
        for i, data in enumerate(dset_vals):
            f[dset_names[i]][existing_examples: total_examples] = data
        return total_examples


def prep_datasets_for_targetz(hdf5file, dset_description, img_dimensions):
    """
    hdf5file - where we will add dsets,
    dset_desciption - ordered dict containing all the pieces of the dset
    img_dimensions - list [x,u,v] of tuples of (w, h)
    """
    dset_names = dset_description.keys()
    if 'hits-x' in dset_names:
        create_view_dset(hdf5file, 'hits-x',
                         img_dimensions[0][0], img_dimensions[0][1])
    if 'hits-u' in dset_names:
        create_view_dset(hdf5file, 'hits-u',
                         img_dimensions[1][0], img_dimensions[1][1])
    if 'hits-v' in dset_names:
        create_view_dset(hdf5file, 'hits-v',
                         img_dimensions[2][0], img_dimensions[2][1])
    if 'segments' in dset_names:
        create_1d_dset(hdf5file, 'segments', 'uint8', 'z-segment')
    if 'zs' in dset_names:
        create_1d_dset(hdf5file, 'zs', 'float32', 'z')
    if 'planecodes' in dset_names:
        create_1d_dset(hdf5file, 'planecodes', 'uint16', 'plane-id-code')
    if 'eventids' in dset_names:
        create_1d_dset(hdf5file, 'eventids', 'uint64',
                       'run+subrun+gate+slices[0]')


def build_nukecc_vtx_study_dset_description(views, img_dimensions):
    """
    views - list or string of 'xuv' views to include
    img_dimensions - a list of tuples of (w,h) - should be final output size
    """
    dset_description = OrderedDict(
        (('hits-x', img_dimensions[0]),
         ('hits-u', img_dimensions[1]),
         ('hits-v', img_dimensions[2]),
         ('segments', ('uint8', 'z-segment')),
         ('zs', ('float32', 'z')),
         ('planecodes', ('uint16', 'plane-id-code')),
         ('eventids', ('uint64', 'run+subrun+gate+slices[0]')))
    )
    if 'x' not in views:
        del dset_description['hits-x']
    if 'u' not in views:
        del dset_description['hits-u']
    if 'v' not in views:
        del dset_description['hits-v']
    return dset_description


def filter_nukecc_vtx_det_vals_for_names(vals, names):
    """
    look through all the dsets we wish to include by name and only keep
    the vals that match that name (by index!) in the return vector.

    note that we are relying on the vals showing up in the same order
    as the names generated from the nukecc dset description generator.
    there is no extra metadata in the list to identify the correct pairing.

    raw nukecc vals structure is like:
        dset_vals = [dataX, dataU, dataV, targs, zs, planecodes, eventids]
    """
    new_vals = []
    full_name_list = build_nukecc_vtx_study_dset_description(
        'xuv', [(0, 0), (0, 0), (0, 0)]).keys()
    for i, name in enumerate(full_name_list):
        if name in names:
            new_vals.append(vals[i])
    return new_vals


def transform_view(dset_vals, view):
        """
        we will _replace_ the original, unshifted values with shifted and
        flipped values. (don't return the unmodified data.) this is because
        we do not want to mix shifted/flipped images with "real" images
        in data files - the shifted/flipped images are for pre-training or
        special types of training.

        we must duplicate every entry of every object in dset_vals and append
        them in sync with the new images created from the view data. this
        function assumes the view data is always in the first entry of the
        dset_vals list and that there is only one entry in the list with
        view data (specified by `view`)
        """
        allowed_trans = ['shift-1', 'shift+1']
        if view == 'x':
            allowed_trans.append('flip')

        viewdata = dset_vals[0]
        restlist = dset_vals[1:]
        new_viewdata = []
        new_restdata = []
        for _ in restlist:
            new_restdata.append([])
        print('Allowed transformations: {}'.format(allowed_trans))
        for i, img in enumerate(viewdata):
            for trans in allowed_trans:
                for j in range(len(restlist)):
                    new_restdata[j].append(restlist[j][i])
                if trans == 'flip':
                    new_img = img[:, ::-1, :]
                    new_restdata[-1][-1] += 50
                else:
                    shift = int(re.search(r'[-+0-9]+', trans).group(0))
                    new_restdata[-1][-1] += (50 + shift)
                    new_img = shift_img_updown(img, shift)
                new_viewdata.append(new_img)

        # put the view data back in front
        new_restdata.insert(0, new_viewdata)

        # TODO: need to update the eventids so they are not all the same

        # in practice, it appears we don't need to be very careful about
        # setting the type of the lists we have built to be np.ndarrays,
        # etc. - lists containing values from the numpy arrays seems to
        # work when passed back into the hdf5 file - perhaps because
        # we've already declared the dtypes for the data sets in the hdf5
        return new_restdata


def make_nukecc_vtx_hdf5_file(imgw, imgh, trims, views,
                              filebase, hdf5file, add_target_padding=False,
                              apply_transforms=False,
                              insert_x_padding_into_uv=True):
    """
    imgw, imgh - ints that specify the image size for `reshape`
    filebase - pattern used for files to match into the output
    hdf5file - name of the output file

    NOTE: trims [x,u,v] are for AFTER padding numbers!

    note that imgw traverses the "y" direction and imgh traverses the "x"
    direction in the classic mathematician's graph

    note that filebase is a pattern - if multiple files match
    the pattern, then multiple files will be included in the
    single output file
    """
    print('Making hdf5 file for img-in x: {} x {} and out {} x {}-{}'.format(
        imgw, imgh, imgw, trims[0][0], trims[0][1]))
    print('Making hdf5 file for img-in u: {} x {} and out {} x {}-{}'.format(
        imgw, imgh, imgw, trims[1][0], trims[1][1]))
    print('Making hdf5 file for img-in v: {} x {} and out {} x {}-{}'.format(
        imgw, imgh, imgw, trims[2][0], trims[2][1]))

    files = make_file_list(filebase)
    f = prepare_hdf5_file(hdf5file)

    img_dims = [(imgw, trims[0][1] - trims[0][0]),
                (imgw, trims[1][1] - trims[1][0]),
                (imgw, trims[2][1] - trims[2][0])]
    dset_description = build_nukecc_vtx_study_dset_description(views, img_dims)
    print(dset_description)
    prep_datasets_for_targetz(f, dset_description, img_dims)
    dset_names = dset_description.keys()

    total_examples = 0

    for fname in files:
        print("Iterating over file:", fname)
        dataX, dataU, dataV, targs, zs, planecodes, eventids = \
            get_nukecc_vtx_study_data_from_file(
                fname, imgw, imgh, trims, add_target_padding,
                insert_x_padding_into_uv)
        print('data shapes:',
              np.shape(dataX), np.shape(dataU), np.shape(dataV))
        dset_vals = [dataX, dataU, dataV, targs, zs, planecodes, eventids]
        dset_vals = filter_nukecc_vtx_det_vals_for_names(dset_vals, dset_names)
        if len(views) == 1 and apply_transforms:
            dset_vals = transform_view(dset_vals, views[0])
        total_examples = add_data_to_hdf5file(f, dset_names, dset_vals)

    add_split_dict(f, dset_names, total_examples)

    f.close()


if __name__ == '__main__':

    f = 'minosmatch_hadmult_me1Bmc_tiny_0000.dat.gz'
    hadron_data = get_hadmult_study_data_from_file(f)

    # from optparse import OptionParser
    # parser = OptionParser(usage=__doc__)
    # parser.add_option('-a', '--apply_transforms', default=False,
    #                   dest='apply_transforms', help='Apply image transforms',
    #                   metavar='APPLY_IMG_TRANS', action='store_true')
    # parser.add_option('-b', '--basename', default='nukecc_skim_me1Bmc',
    #                   help='Input files base name', metavar='BASE_NAME',
    #                   dest='filebase')
    # parser.add_option('-c', '--check_target_padding', default=False,
    #                   dest='check_target_padding', help='Check target padding',
    #                   metavar='CHECK_TARG_PAD', action='store_true')
    # parser.add_option('-o', '--output', default='minerva_fuel.hdf5',
    #                   help='Output filename', metavar='OUTPUT_NAME',
    #                   dest='hdf5file')
    # parser.add_option('-p', '--padded_targets', default=False,
    #                   dest='padding', help='Include target padding',
    #                   metavar='TARG_PAD', action='store_true')
    # parser.add_option('-s', '--skim', default='nukecc_vtx',
    #                   help='Skimmed sample type', metavar='SKIM',
    #                   dest='skim')
    # parser.add_option('-t', '--inp_height', default=94, type='int',
    #                   help='Image input height', metavar='IMG_HEIGHT',
    #                   dest='imgh')
    # parser.add_option('-v', '--views', default='xuv', dest='views',
    #                   help='Views (xuv)', metavar='VIEWS')
    # parser.add_option('-w', '--inp_width', default=127, type='int',
    #                   help='Image input width', metavar='IMG_WIDTH',
    #                   dest='imgw')
    # parser.add_option('-x', '--remove-xpaduv', default=True,
    #                   help='Insert x padding in u/v', metavar='XPAD_UV',
    #                   dest='insert_x_padding_into_uv', action='store_false')
    # parser.add_option('--trim_column_down_x', default=94, type='int',
    #                   help='Trim column downstream x', metavar='XTRIM_COL_DN',
    #                   dest='trim_column_down_x')
    # parser.add_option('--trim_column_up_x', default=0, type='int',
    #                   help='Trim column upstream x', metavar='XTRIM_COL_UP',
    #                   dest='trim_column_up_x')
    # parser.add_option('--trim_column_down_uv', default=94, type='int',
    #                   help='Trim column downstream uv',
    #                   metavar='UVTRIM_COL_DN',
    #                   dest='trim_column_down_uv')
    # parser.add_option('--trim_column_up_uv', default=0, type='int',
    #                   help='Trim column upstream uv',
    #                   metavar='UVTRIM_COL_UP',
    #                   dest='trim_column_up_uv')
    # (options, args) = parser.parse_args()

    # if options.check_target_padding:
    #     padding = get_total_target_padding()
    #     print("Total target padding is {} columns.".format(padding))
    #     sys.exit(0)

    # allowed_views = list('xuv')
    # views = list(options.views.lower())
    # for v in views:
    #     if v not in allowed_views:
    #         print('{} is not an allowed view option.'.format(v))
    #         print("Please use any/all of 'xuv' (case insensitive).")
    #         sys.exit(1)
    # filebase = options.filebase
    # hdf5file = options.hdf5file

    # apply_trans = options.apply_transforms
    # if apply_trans and len(views) > 1:
    #     print('Only apply image transforms to one-view files.')
    #     print('Please re-run with views == x, u, OR v.')
    #     sys.exit(1)

    # if options.padding:
    #     padding = get_total_target_padding()
    #     print("Adding {} padding columns for the passive targets...".format(
    #         padding))
    #     print("  Please note that target padding must be included by hand.")
    #     if not options.insert_x_padding_into_uv:
    #         print("Cannot have target padding and no x padding for u/v.")
    #         print("Target padding insertion math is off at that point.")
    #         sys.exit(1)

    # xtrims = (options.trim_column_up_x, options.trim_column_down_x)
    # utrims = (options.trim_column_up_uv, options.trim_column_down_uv)
    # vtrims = (options.trim_column_up_uv, options.trim_column_down_uv)
    # trims = [xtrims, utrims, vtrims]

    # if options.skim == 'nukecc_vtx':
    #     make_nukecc_vtx_hdf5_file(options.imgw, options.imgh, trims,
    #                               views, filebase, hdf5file, options.padding,
    #                               apply_trans,
    #                               options.insert_x_padding_into_uv)
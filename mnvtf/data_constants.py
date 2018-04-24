import tensorflow as tf

EVENT_DATA = 'event_data'
EVENTIDS = 'eventids'
PLANECODES = 'planecodes'
VTX_DATA = 'vtx_data'
SEGMENTS = 'segments'
ZS = 'zs'
GEN_DATA = 'gen_data'
QSQRD = 'Q2'
WINV = 'W'
XBJ = 'x'
YBJ = 'y'
CURRENT = 'current'
INT_TYPE = 'int_type'
TARGETZ = 'targetZ'
HADRO_DATA = 'hadro_data'
ESUM_CHGDKAONS = 'esum_chgdkaons'
ESUM_CHGDPIONS = 'esum_chgdpions'
ESUM_HADMULTMEAS = 'esum_hadmultmeas'
ESUM_NEUTPIONS = 'esum_neutpions'
ESUM_NEUTRONS = 'esum_neutrons'
ESUM_OTHERS = 'esum_others'
ESUM_PROTONS = 'esum_protons'
N_CHGDKAONS = 'n_chgdkaons'
N_CHGDPIONS = 'n_chgdpions'
N_HADMULTMEAS = 'n_hadmultmeas'
N_NEUTPIONS = 'n_neutpions'
N_NEUTRONS = 'n_neutrons'
N_OTHERS = 'n_others'
N_PROTONS = 'n_protons'
IMG_DATA = 'img_data'
HITIMESU = 'hitimes-u'
HITIMESV = 'hitimes-v'
HITIMESX = 'hitimes-x'
PID_DATA = 'pid_data'
PIDU = 'pid-u'
PIDV = 'pid-v'
PIDX = 'pid-x'
LEPTO_DATA = 'lepto_data'
ESUM_ELECTRONS = 'esum_electrons'
ESUM_MUONS = 'esum_muons'
ESUM_TAUS = 'esum_taus'
N_ELECTRONS = 'n_electrons'
N_MUONS = 'n_muons'
N_TAUS = 'n_taus'

# groups in the HDF5 file marked for translation to TFRecord; ultimately, we
# can't build this list dynamically because there is no way to get it from
# the TFRecord file by inspection - we must unpack with foreknowledge of
# what is there to parse the raw bytes.
HADMULTKINE_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    GEN_DATA: [QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ],
    HADRO_DATA: [
        ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
        ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
        N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS, N_NEUTPIONS,
        N_NEUTRONS, N_OTHERS, N_PROTONS
    ],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX],
    LEPTO_DATA: [
        ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
        N_ELECTRONS, N_MUONS, N_TAUS
    ]
}
HADMULTKINE_TYPE = 'hadmultkineimgs'

VTXFINDING_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX]
}
VTXFINDING_TYPE = 'vtxfndingimgs'

IMGING_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX]
}
IMGING_TYPE = 'mnvimgs'

SEGMENTATION_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX],
    PID_DATA: [PIDU, PIDV, PIDX]
}
SEGMENTATION_TYPE = "segmentation"

VALID_SET_OF_GROUPS = set(
    tuple(HADMULTKINE_GROUPS_DICT.keys()) +
    tuple(VTXFINDING_GROUPS_DICT.keys()) +
    tuple(IMGING_GROUPS_DICT.keys()) +
    tuple(SEGMENTATION_GROUPS_DICT.keys())
)


def make_mnv_data_dict(list_of_groups):
    """
    create a dict of fields to extract from the hdf5 with target dtypes.
    """
    # eventids are really (in the hdf5) uint64, planecodes are really uint16;
    # use tf.{int64,int32,uint8} because these are the dtypes that one-hot
    # supports (_not_ int16 or uint16, at least in TF v1.2); use int64 instead
    # of unit64 because reshape supports int64 (and not uint64).
    data_list = []
    for g in list_of_groups:
        if g in VALID_SET_OF_GROUPS:
            if g == EVENT_DATA:
                data_list.extend([
                    (EVENTIDS, tf.int64, g),
                ])
            if g == VTX_DATA:
                data_list.extend([
                    (PLANECODES, tf.int32, g),
                    (SEGMENTS, tf.uint8, g),
                    (ZS, tf.float32, g),
                ])
            if g == IMG_DATA:
                data_list.extend([
                    (HITIMESU, tf.float32, g),
                    (HITIMESV, tf.float32, g),
                    (HITIMESX, tf.float32, g),
                ])
            if g == GEN_DATA:
                data_list.extend([
                    (QSQRD, tf.float32, g),
                    (WINV, tf.float32, g),
                    (XBJ, tf.float32, g),
                    (YBJ, tf.float32, g),
                    (CURRENT, tf.int32, g),
                    (INT_TYPE, tf.int32, g),
                    (TARGETZ, tf.int32, g),
                ])
            if g == HADRO_DATA:
                data_list.extend([
                    (ESUM_CHGDKAONS, tf.float32, g),
                    (ESUM_CHGDPIONS, tf.float32, g),
                    (ESUM_HADMULTMEAS, tf.float32, g),
                    (ESUM_NEUTPIONS, tf.float32, g),
                    (ESUM_NEUTRONS, tf.float32, g),
                    (ESUM_OTHERS, tf.float32, g),
                    (ESUM_PROTONS, tf.float32, g),
                    (N_CHGDKAONS, tf.int32, g),
                    (N_CHGDPIONS, tf.int32, g),
                    (N_HADMULTMEAS, tf.int32, g),
                    (N_NEUTPIONS, tf.int32, g),
                    (N_NEUTRONS, tf.int32, g),
                    (N_OTHERS, tf.int32, g),
                    (N_PROTONS, tf.int32, g),
                ])
            if g == LEPTO_DATA:
                data_list.extend([
                    (ESUM_ELECTRONS, tf.float32, g),
                    (ESUM_MUONS, tf.float32, g),
                    (ESUM_TAUS, tf.float32, g),
                    (N_ELECTRONS, tf.int32, g),
                    (N_MUONS, tf.int32, g),
                    (N_TAUS, tf.int32, g),
                ])
        else:
            raise ValueError('Unrecognized group')
    mnv_data = {}
    for datum in data_list:
        mnv_data[datum[0]] = {}
        mnv_data[datum[0]]['dtype'] = datum[1]
        mnv_data[datum[0]]['byte_data'] = None
        mnv_data[datum[0]]['group'] = datum[2]

    return mnv_data

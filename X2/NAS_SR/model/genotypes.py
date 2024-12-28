from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_multi = namedtuple('Genotype', 'normal_bottom normal_concat_bottom upsampling_bottom upsampling_concat_bottom \
                                         normal_mid normal_concat_mid upsampling_mid upsampling_concat_mid \
                                         normal_top normal_concat_top')
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

LSTM_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


COMPACT_PRIMITIVES = [
    'rcab',
    'w1',
    'w2',
    'w3',
    'w4',
    # 'resb',
]


ATT_PRIMITIVES = [
    # 'SE_Block',
    'CALayer',
    'ESA',
    'eca_layer',
    'skip_connect',
]


COMPACT_PRIMITIVES_UPSAMPLING = [
    'sub_pixel',
    'deconvolution',
    'bilinear',
    'nearest',
    'area',
]


PRUNER_PRIMITIVES = [
    'none',
    'same',
    'skip_connect',
]


genotype = [Genotype(normal=[('w3', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w4', 0)], normal_concat=range(1, 2), reduce=[('CALayer', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w3', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w1', 0)], normal_concat=range(1, 2), reduce=[('CALayer', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w1', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w4', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w3', 0)], normal_concat=range(1, 2), reduce=[('eca_layer', 0)], reduce_concat=range(1, 2)), Genotype(normal=[('w4', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2))]
channel = [[22, 5], [4, 21], [20, 6], [15, 5], [10, 4], [4, 4], [18, 6], [6, 18]]

genotype_ours_b = [Genotype(normal=[('w3', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2)),
                   Genotype(normal=[('w3', 0)], normal_concat=range(1, 2), reduce=[('CALayer', 0)], reduce_concat=range(1, 2))]
channel_ours_b = [[16, 5], [9, 4]]

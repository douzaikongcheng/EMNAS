import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/pengzhangheng/data',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/800-800',
                    help='train_w/train_controller data range')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--noise_std', type=float, default=30/255,
                    help='noise std')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
# parser.add_argument('--model', default='EDSR',
parser.add_argument('--model', default='model_sr',
                    help='model name')


parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
# parser.add_argument('--pre_train', type=str, default='../experiment/retain_darts_30/model/model_best.pt',
                    help='pre-trained model directory')

parser.add_argument('--pre_train_step1', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--stage', type=str, default='step1',
                    help='loss function stage')
# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
# parser.add_argument('--lr', type=float, default=1e-4,
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='0.5',
                    help='learning rate decay type')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--lr_decay', type=int, default=100,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='Mstep_300_450_600',
                    help='learning rate decay type')
# parser.add_argument('--decay_type', type=str, default='step',
#                     help='learning rate decay type: MultiStep | cosine')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='retain_darts_30_test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--teacher_model_weight_path', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
# parser.add_argument('--print_every', type=int, default=200,
parser.add_argument('--print_every', type=int, default=40,
                    help='how many batches to wait before logging training status')
parser.add_argument('--repeat_data_time', type=int, default=1,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', type=int,default=1,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')


parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')

# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_len', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')


parser.add_argument('--init_channels', type=int, default=64, help='number of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
# parser.add_argument('--channel_config', type=str, default='[[320, 64], [320, 64], [320, 64], [320, 64], [320, 64], [320, 64]]', help='number of init channels')
# parser.add_argument('--channel_config', type=str, default='[[20, 4], [20, 4], [20, 4], [20, 4], [20, 4], [20, 4]]', help='number of init channels')
# parser.add_argument('--channel_config', type=str, default='[[128, 32], [128, 32], [128, 32], [128, 32], [128, 32], [128, 32]]', help='number of init channels')
parser.add_argument('--channel_config', type=str, default='[[64, 8], [64, 8], [64, 8], [64, 8], [64, 8], [64, 8]]', help='number of init channels')
parser.add_argument('--nodes', type=int, default=1, help='total number of nodes')
parser.add_argument('--genotype', type=str, default='genotype', help='which architecture to use')
parser.add_argument('--geno_channel', type=str, default='channel', help='which architecture to use')
parser.add_argument('--arch_start_training', type=int, default=1, help='Epoch that the training of controller starts')
parser.add_argument('--sampling', type=int, default=1, help='when test and save the arch, the number to sample')
parser.add_argument('--sampling_epoch_margin', type=int, default=1, help='the margin of epoch to sample')
parser.add_argument('--tail_fea', type=int, default=8, help='pretrain epoch')
parser.add_argument('--search_n_feats', type=int, default=40, help='pretrain epoch')
parser.add_argument('--offset', type=int, default=6, help='pretrain epoch')

parser.add_argument('--sparse', type=float, default=1e-5, help='weight decay for arch encoding')
parser.add_argument('--percent_ir', type=float, default=0.8, help='weight decay for arch encoding')





# add CSD
parser.add_argument('--neg_num', type=int, default=8, help='negative samples number')
parser.add_argument('--t_lambda', type=float, default=1, help='weight of l1(hr, teacher_sr)')
parser.add_argument('--contra_lambda', type=float, default=200, help='weight of contra_loss')
parser.add_argument('--ad_lambda', type=float, default=0, help='weight of adversarial loss')
parser.add_argument('--percep_lambda', type=float, default=0, help='weight of perceptual loss')
parser.add_argument('--kd_lambda', type=float, default=0, help='weight of kd loss')
parser.add_argument('--vgg_weight', nargs='+', type=float, default=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
                    help='weight of vgg features in contrastive loss')
parser.add_argument('--d_func', type=str, default="L1", help='the distance function in contrastive loss')
parser.add_argument('--mean_outside', action='store_true', help='calc mean for negative samples outside the contrast_loss')
parser.add_argument('--t_l_remove', type=int, default=0, help='remove teacher loss @ epoch {t_l_remove}')
parser.add_argument('--contrast_t_detach', default=1, help='detach teacher in contrast_loss')
parser.add_argument('--gt_as_pos', action='store_true', help='use gt as positive sample')
parser.add_argument('--blur_sigma', type=float, default=0, help='blur sigma of neg sample')
parser.add_argument('--noise_sigma', type=float, default=0, help='noise sigma of neg sample')






args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
# print(args.scale)
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

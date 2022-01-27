import argparse
import os
import time
from pathlib import Path

class DebugOptions:
    def __init__(self):

        self.dataset="mongoose"
        self.datafile="/home/vsivan/t2_list.txt"
        self.gpu_ids='0'
        self.checkpoint_dir='./checkpoints'
        self.load_model_dir=""
        self.phase="train"
        self.batch_size=2
        self.random_crop=1
        self.random_mask=1
        self.mask_type='rect'
        self.pretrain_network=0
        self.lambda_adv = 1e-3
        self.lambda_rec=1.4
        self.lambda_ae=1.2
        self.lambda_mrf=0.05
        self.lambda_gp=10
        self.random_seed=False
        self.padding='SAME'
        self.D_max_iters=5
        self.lr=1e-5
        self.train_spe=1000
        self.epochs=40
        self.viz_steps=5
        self.spectral_norm=1
        self.img_shapes=(16,32,32)
        self.mask_shapes=tuple([i//2 for i in self.img_shapes])
        self.max_delta_shapes=(32,32)
        self.margins=(0,0)
        self.g_cnum=32
        self.d_cnum=64
        self.vgg19_path='vgg19_weights/imagenet-vgg-verydeep-19.mat'
        self.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.model_name = 'GMCNN'
        self.model_folder = self.date_str + '_' + self.model_name
        self.model_folder += '_' + self.dataset
        self.model_folder += '_b' + str(self.batch_size)
        self.model_folder += '_s' + str(self.img_shapes[0]) + 'x' + str(self.img_shapes[1])
        self.model_folder += '_gc' + str(self.g_cnum)
        self.model_folder += '_dc' + str(self.d_cnum)

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--dataset', type=str, default='paris_streetview',
                                 help='dataset of the experiment.')
        self.parser.add_argument('--data_file', type=Path, default='', help='the file storing training image paths')
        self.parser.add_argument('--root_dir', type=Path, default=None, help='the root directory of the training files')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2')
        self.parser.add_argument('--checkpoint_dir', type=Path, default=None, help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='', help='pretrained models are given here')
        self.parser.add_argument('--phase', type=str, default='train')

        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')

        # for setting inputs
        self.parser.add_argument('--random_crop', type=int, default=1,
                                 help='using random crop to process input image when '
                                      'the required size is smaller than the given size')
        self.parser.add_argument('--random_mask', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='rect')
        self.parser.add_argument('--pad_mode', type=str)
        self.parser.add_argument('--pretrain_network', type=int, default=1)
        self.parser.add_argument('--lambda_adv', type=float, default=1e-3)
        self.parser.add_argument('--lambda_rec', type=float, default=1.4)
        self.parser.add_argument('--lambda_ae', type=float, default=1.2)
        self.parser.add_argument('--lambda_mrf', type=float, default=0.05)
        self.parser.add_argument('--lambda_gp', type=float, default=10)
        self.parser.add_argument('--random_seed', type=bool, default=False)
        self.parser.add_argument('--padding', type=str, default='SAME')
        self.parser.add_argument('--D_max_iters', type=int, default=5)
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')

        self.parser.add_argument('--train_spe', type=int, default=1000)
        self.parser.add_argument('--pretrain_epochs', type=int, default=40)
        self.parser.add_argument('--finetune_epochs', type=int, default=40)
        self.parser.add_argument('--viz_steps', type=int, default=5)
        self.parser.add_argument('--spectral_norm', type=int, default=1)

        self.parser.add_argument('--img_shapes', type=str, default='32,64,64',
                                 help='given shape parameters: d,h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='16,32,32',
                                 help='given mask parameters: d,h,w')
        self.parser.add_argument('--max_delta_shapes', type=str, default='32,32')
        self.parser.add_argument('--margins', type=str, default='0,0')


        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')

        # for id-mrf computation
        self.parser.add_argument('--vgg19_path', type=str, default='vgg19_weights/imagenet-vgg-verydeep-19.mat')
        # for instance-wise features
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        paths_that_should_exist = (self.opt.data_file, self.opt.root_dir)
        for p in paths_that_should_exist:
            assert os.path.exists(p)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))

        assert self.opt.random_crop in [0, 1]
        self.opt.random_crop = True if self.opt.random_crop == 1 else False

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.pretrain_network in [0, 1]
        self.opt.pretrain_network = True if self.opt.pretrain_network == 1 else False

        assert self.opt.spectral_norm in [0, 1]
        self.opt.spectral_norm = True if self.opt.spectral_norm == 1 else False

        assert self.opt.padding in ['SAME', 'MIRROR']

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        str_max_delta_shapes = self.opt.max_delta_shapes.split(',')
        self.opt.max_delta_shapes = [int(x) for x in str_max_delta_shapes]

        str_margins = self.opt.margins.split(',')
        self.opt.margins = [int(x) for x in str_margins]

        # NOTE: we call the directory we want to save stuff in the model_dir
        # as this enables us to use the same variable name in our scripts for
        # both test and train
        assert self.opt.checkpoint_dir is not None
        self.opt.model_folder = self.opt.checkpoint_dir
        os.makedirs(self.opt.checkpoint_dir, exist_ok=True)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

import argparse
import os
import time
from pathlib import Path

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='inpainting',
                                 help='The dataset of the experiment.')
        self.parser.add_argument('--data_file', type=Path, help='the file storing testing file names')
        self.parser.add_argument('--root_dir', type=Path, help="the root directory")
        self.parser.add_argument('--test_dir', type=Path, default='./test_results', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=Path, help='directory corresponding to a trained model')
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--scratch', action='store_true', default=False)
        self.parser.add_argument('--saving_path', type=Path)

        self.parser.add_argument('--model', type=str, default='gmcnn')
        self.parser.add_argument('--random_mask', type=int, default=0,
                                 help='using random mask')

        self.parser.add_argument('--img_shapes', type=str, default='32,64,64',
                                 help='given shape parameters: d,h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='16,32,32',
                                 help='given mask parameters: d,h,w')
        self.parser.add_argument('--mask_type', type=str, default='rect')
        self.parser.add_argument('--mode', type=str, default='save')
        self.parser.add_argument('--phase', type=str, default='test')

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=32,
                                 help='# of discriminator filters in first conv layer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        paths_that_should_exist = (
            self.opt.data_file, self.opt.root_dir, self.opt.load_model_dir
        )
        if self.opt.scratch:
            paths_that_should_exist = paths_that_should_exist[:-1]

        for p in paths_that_should_exist:
            assert os.path.exists(p)


        os.makedirs(self.opt.test_dir, exist_ok=True)

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        # model name and date
        self.opt.date_str = 'test_'+time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.dataset + '_' + self.opt.model
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        if self.opt.random_mask:
            self.opt.model_folder += '_seed-' + str(self.opt.seed)
        if not self.opt.saving_path:
            self.opt.saving_path = self.opt.test_dir/self.opt.model_folder

        if self.opt.mode == 'save':
            os.makedirs(self.opt.saving_path, exist_ok=True)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

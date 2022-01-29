import argparse
import os
import time
from pathlib import Path

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_file', type=Path, help='the file storing testing file names')
        self.parser.add_argument('--root_dir', type=Path, help="the root directory")
        self.parser.add_argument('--load_model_dir', type=Path, help='directory corresponding to a trained model')
        self.parser.add_argument('--saving_path', type=Path)
        self.parser.add_argument('--lambda_rec', type=float, default=1.4)
        self.parser.add_argument('--lambda_ae', type=float, default=1.2)
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')
        self.parser.add_argument('--gpu_ids', type=str, default='0')

        self.parser.add_argument('--img_shapes', type=str, default='32,64,64',
                                 help='given shape parameters: d,h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='16,32,32',
                                 help='given mask parameters: d,h,w')

        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        paths_that_should_exist = (
            self.opt.data_file, self.opt.root_dir, self.opt.load_model_dir,
        )

        for p in paths_that_should_exist:
            assert os.path.exists(p)

        os.makedirs(self.opt.saving_path, exist_ok=True)

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        os.makedirs(self.opt.saving_path, exist_ok=True)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

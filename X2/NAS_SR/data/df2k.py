# import os
# from data import srdata
#
# class DF2K(srdata.SRData):
#     def __init__(self, args, name='DF2K', train=True, benchmark=False):
#         super(DF2K, self).__init__(
#             args, name=name, train=train, benchmark=benchmark
#         )
#
#     def _set_filesystem(self, dir_data):
#         super(DF2K, self)._set_filesystem(dir_data)
#         self.dir_hr = os.path.join(self.apath, 'DF2K_HR')
#         self.dir_lr = os.path.join(self.apath, 'DF2K_LR_bicubic')
#
#
import os
from data import srdata

class DF2K(srdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DF2K, self).__init__(
            # args, name=name, train=train, benchmark=benchmark,train_controller=args.train_controller
            args, name=name, train=True, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DF2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        # print(names_hr)
        # print(names_lr)
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DF2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DF2K_HR')
        self.dir_lr = os.path.join(self.apath, 'DF2K_LR_bicubic')
        # print(self.dir_hr)
        # print(self.dir_lr)
        if self.input_large: self.dir_lr += 'L'




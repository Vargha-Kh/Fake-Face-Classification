import os
import shutil
import pandas as pd


class PreparingDatasets:
    def __init__(self, framework):
        self.framework = framework

    def transfer_directory_items(self, in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False):
        print(f'starting to copying/moving from {in_dir} to {out_dir}')
        if remove_out_dir or os.path.isdir(out_dir):
            os.system(f'rm -rf {out_dir}; mkdir -p {out_dir}')
        if mode == 'cp':
            for name in transfer_list:
                shutil.copy(os.path.join(in_dir, name), out_dir)
        elif mode == 'mv':
            for name in transfer_list:
                shutil.move(os.path.join(in_dir, name), out_dir)
        else:
            raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
        print(f'finished copying/moving from {in_dir} to {out_dir}')

    def dir_train_test_split(self, in_dir, out_dir, test_size=0.3, result_names=('train', 'val'), mode='cp',
                             remove_out_dir=False):
        from sklearn.model_selection import train_test_split
        list_ = os.listdir(in_dir)
        train_name, val_name = train_test_split(list_, test_size=test_size)
        self.transfer_directory_items(in_dir,
                                      os.path.join(out_dir, result_names[0]),
                                      train_name, mode=mode,
                                      remove_out_dir=remove_out_dir)
        self.transfer_directory_items(in_dir,
                                      os.path.join(out_dir, result_names[1]),
                                      val_name, mode=mode,
                                      remove_out_dir=remove_out_dir)

    def create_directory(self):
        if self.framework == 'pytorch':
            try:
                os.makedirs('./fake_real-faces/train/real')
                os.makedirs('./fake_real-faces/train/fake')
                os.makedirs('./fake_real-faces/val/real')
                os.makedirs('./fake_real-faces/val/fake')
                os.makedirs('./fake_real-faces/test/real')
                os.makedirs('./fake_real-faces/test/fake')
                os.makedirs('./data_splitting/fake/train')
                os.makedirs('./data_splitting/fake/val')
                os.makedirs('./data_splitting/real/train')
                os.makedirs('./data_splitting/real/val')
            except:
                pass
        else:
            print("Wrong Type of Framework!")

    def preparing_datasets(self, split_size=0.3):
        if self.framework == 'pytorch':
            self.create_directory()
            self.dir_train_test_split('./fake_real-faces/real/',
                                      './data_splitting/real/', test_size=split_size)
            self.dir_train_test_split('./fake_real-faces/fake/',
                                      './data_splitting/fake/', test_size=split_size)

            # for train data
            # Real
            train_real = './fake_real-faces/train/real/'
            source_train = "./data_splitting/real/train"
            for filename in os.listdir(source_train):
                try:
                    path = os.path.join(source_train, filename)
                    shutil.copy(path, train_real)
                    print("File ", filename, " Successfully copied to ", train_real)
                except FileNotFoundError:
                    pass

            # Fake
            train_fake = './fake_real-faces/train/fake/'
            source_train = "./data_splitting/fake/train"
            for filename in os.listdir(source_train):
                try:
                    path = os.path.join(source_train, filename)
                    shutil.copy(path, train_fake)
                    print("File ", filename, " Successfully copied to ", train_fake)
                except FileNotFoundError:
                    pass


            # For validation data
            # real
            val_real = './fake_real-faces/val/real/'
            source_val = "./data_splitting/real/val"
            for filename in os.listdir(source_val):
                try:
                    path = os.path.join(source_val, filename)
                    shutil.copy(path, val_real)
                    print("File ", filename, " Successfully copied to ", val_real)
                except FileNotFoundError:
                    pass

            # fake
            val_fake = './fake_real-faces/val/fake/'
            source_val = "./data_splitting/fake/val"
            for filename in os.listdir(source_val):
                try:
                    path = os.path.join(source_val, filename)
                    shutil.copy(path, val_fake)
                    print("File ", filename, " Successfully copied to ", val_fake)
                except FileNotFoundError:
                    pass

            # for test data
            # test_pnemonia = './covid-19/test/covid/'
            # source_test = "./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
            #
            # for i in move_test_pnemonia:
            #     path2 = os.path.join(source_test, i)
            #     shutil.copy(path2, test_pnemonia)
            #     print("File ", i, " Successfully copied to ", test_pnemonia)
            #
            # test_normal = './covid-19/test/normal/'
            # move_test_normal = test_data[test_data.Label == 'Normal']['X_ray_image_name'].values
            # for i in move_test_normal:
            #     path3 = os.path.join(source_test, i)
            #     shutil.copy(path3, test_normal)
            #     print("File ", i, " Successfully copied to ", test_normal)

        else:
            print("Wrong Type of Framework")

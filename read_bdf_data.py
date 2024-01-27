# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import glob
import os
import pickle
from scipy.signal import resample
import numpy as np

from neuracle_lib.readbdfdata import readbdfdata


# Versions:
# 	v0.1: 2018-08-14, orignal
# Author: FANG Junying, fangjunying@neuracle.cn
# Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/


def check_files_format(path):
    filename = []
    pathname = []
    if len(path) == 0:
        raise TypeError('please select valid file')

    elif len(path) == 1:
        (temppathname, tempfilename) = os.path.split(path[0])
        if 'edf' in tempfilename:
            filename.append(tempfilename)
            pathname.append(temppathname)
            return filename, pathname
        elif 'bdf' in tempfilename:
            raise TypeError('unsupport only one neuracle-bdf file')
        else:
            raise TypeError('not support such file format')

    else:
        temp = []
        temppathname = r''
        evtfile = []
        idx = np.zeros((len(path) - 1,))
        for i, ele in enumerate(path):
            (temppathname, tempfilename) = os.path.split(ele)
            if 'data' in tempfilename:
                temp.append(tempfilename)
                if len(tempfilename.split('.')) > 2:
                    try:
                        idx[i] = (int(tempfilename.split('.')[1]))
                    except:
                        raise TypeError('no such kind file')
                else:
                    idx[i] = 0
            elif 'evt' in tempfilename:
                evtfile.append(tempfilename)

        pathname.append(temppathname)
        datafile = [temp[i] for i in np.argsort(idx)]

        if len(evtfile) == 0:
            raise TypeError('not found evt.bdf file')

        if len(datafile) == 0:
            raise TypeError('not found data.bdf file')
        elif len(datafile) > 1:
            print('current readbdfdata() only support continue one data.bdf ')
            return filename, pathname
        else:
            filename.append(datafile[0])
            filename.append(evtfile[0])
            return filename, pathname


def process_data(i, eeg, eeg_data, length):
    temp = np.array(eeg_data['data'][i])
    data_1 = temp[eeg['events'][0][0]: eeg['events'][1][0]][-length:]
    data_2 = temp[eeg['events'][2][0]: eeg['events'][3][0]][-length:]
    data_3 = temp[eeg['events'][4][0]: eeg['events'][5][0]][-length:]
    data_4 = temp[eeg['events'][6][0]: eeg['events'][7][0]][-length:]
    data_5 = temp[eeg['events'][8][0]: eeg['events'][9][0]][-length:]
    row_data = np.stack((data_1, data_2, data_3, data_4, data_5))
    return row_data


def downsample_data(data, target_length):
    downsampled_data = resample(data, target_length, axis=-1)
    return downsampled_data

    return downsampled_data


def get_filenames_in_folder(folder_path):
    filenames = glob.glob(os.path.join(folder_path, '*'))
    filenames = [os.path.basename(filename) for filename in filenames]
    # 将文件名按照组号排序
    filenames.sort(key=lambda x: int(x.split('-')[0]))
    # 再将文件名按照情绪排序
    if len(filenames) == 4:
        filenames.sort(key=lambda x: x.split('-')[1])
    return filenames


if __name__ == '__main__':
    # root = Tk()
    # root.withdraw()
    # # select bdf or edf file
    # path = filedialog.askopenfilenames(initialdir='/', title='Select two bdf files',
    #                                    filetypes=(("two bdf files", "*.bdf"), ("one edf files", "*.edf")))
    # # check files format
    # filename, pathname = check_files_format(path)
    # parse data
    filename = ['data.bdf', 'evt.bdf']
    # 更改为自己的路径
    pre_filepath = '/Users/HP/Desktop/脑电/data_origin/'  # change to your own path
    length = 59900
    labels = np.array(
        [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1],
         [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    hvha_data = np.empty((5, 32, length))
    hvla_data = np.empty((5, 32, length))
    lvha_data = np.empty((5, 32, length))
    lvla_data = np.empty((5, 32, length))

    for i in range(len(get_filenames_in_folder(pre_filepath))):
        pre_path_name = pre_filepath + str(get_filenames_in_folder(pre_filepath)[i])
        files = get_filenames_in_folder(pre_path_name)
        prefix = files[0].split("-")  # 判断前缀
        prefix_number = int(prefix[0])
        if prefix_number == 21:  # 跳过第21组
            continue
        for file in files:
            pathname = [pre_path_name + '/' + file]
            eeg = readbdfdata(filename, pathname)
            eeg_data = dict(data=[], ch_names=[])
            eeg_channels = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1",
                            "Oz",
                            "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4",
                            "P8",
                            "PO4", "O2"]
            for j in range(len(eeg['ch_names'])):
                if eeg['ch_names'][j] in eeg_channels:
                    eeg_data['ch_names'].append(eeg['ch_names'][j])
                    eeg_data['data'].append(eeg['data'][j])

            if len(eeg['events']) != 10:
                raise TypeError('not found 10 events in:  ', pathname, len(eeg['events']))

            data = np.empty((5, 32, length))
            for j in range(len(eeg_data['ch_names'])):
                data[:, j, :] = process_data(j, eeg, eeg_data, length)

            emotion = pathname[len(pathname) - 1].split('/')[-1].split('-')[1]
            if emotion == 'hvha':
                hvha_data = data
            elif emotion == 'hvla':
                hvla_data = data
            elif emotion == 'lvha':
                lvha_data = data
            elif emotion == 'lvla':
                lvla_data = data
            else:
                raise TypeError('not found such emotion: ', emotion)

        data = np.concatenate((hvha_data, hvla_data, lvha_data, lvla_data), axis=0)
        # 将data的第三列降采样
        print(data.shape)
        # data = downsample_data(data, 7667)
        # 生成打乱顺序的索引
        random_indices = np.random.permutation(len(data))
        shuffled_data = data[random_indices]
        shuffled_label = labels[random_indices]
        print(shuffled_data.shape)
        print(shuffled_label)
        data_array = {
            'data': shuffled_data,
            'labels': shuffled_label
        }
        new_file_name = f"sample_{i + 1}.dat"
        print(f"正在保存 {new_file_name} ...")
        saved_path = r'/Users/HP/Desktop/脑电/data_net/'
        temp_save_path = os.path.join(saved_path, new_file_name)
        # 保存数据到 .dat 文件
        with open(temp_save_path, 'wb') as file:
            pickle.dump(data_array, file)

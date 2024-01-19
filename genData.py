import time
import time
import random

import numpy
import numpy as np
import os
from PIL import  Image
# from . import getConv
import getConv
import shutil

def get_data_3d(path_str,rates_list,days_list,size=720,wirtetofile_bool=False,writepath_str="./",writefilename_str="None",diameter_int=1,min=0,max=720,number_int=100000):
    '''

    :param path_str: 数据根目录 For example：D:\download\work\work
    :param rates_list:希望采样数据包含的 水灰比 数据列表
    :param days_list:希望采样数据包含的 天数 数据列表
    :param size:采样数据块儿的大小（例如水泥块儿大小720*720*720，size=720）
    :param wirtetofile_bool:随机采样数据是否写入文件文件
    :param writepath_str:随机采样数据写入地址
    :param writefilename_str:随机采样数据写入的文件名
    :param radius_int:三维采样半径，如d=1，直径为2d+1，3*3*3的小块儿
    :param min:采样开始范围的最小点
    :param max:采样开始范围的最大点
    :param number_int:如果写入文件，随机采样点的采样次数
    :return:
    '''
    if len(days_list) < 2:
        print("The number of days must be at least two days or more")
        return  None
    imageDataSize = np.empty([len(rates_list),len(days_list),size,size,size],dtype=int)
    for rate_number in range(len(rates_list)):
        for day_number in range(len(days_list)):
            # D:\download\hw\work\A35
            path = os.path.join(path_str, rates_list[rate_number])

            # D:\download\hw\work\A35\A35
            path = os.path.join(path, rates_list[rate_number])

            # D:\download\hw\work\A35\A35\2d
            path = os.path.join(path, days_list[day_number])
            names = os.listdir(path)
            for i in range(len(names)):
                names[i] = os.path.join(path, names[i])
            imageSize = np.empty([size] * 3, dtype=int)


            for i in range(len(names)):
                image = Image.open(names[i])
                imageSize[i] = np.array(image)
            imageDataSize[rate_number][day_number] = imageSize
            # print("{} {} Done!".format(rates_list[rate_number], days_list[day_number]))
        # print(imageDataSize)

    if wirtetofile_bool:
        images_file = os.path.join(writepath_str,"images")
        masks_file = os.path.join(writepath_str,"masks")
        if os.path.exists(images_file):
            shutil.rmtree(images_file)
        os.mkdir(images_file)
        if os.path.exists(masks_file):
            shutil.rmtree(masks_file)
        os.mkdir(masks_file)

        for i in range(number_int):
            rate_random = random.randrange(len(rates_list))
            days_random = random.randrange(len(days_list) - 1)
            x = random.randrange(size)
            y = random.randrange(size)
            z = random.randrange(size)
            image_data_numpy = getConv.get_data(imageDataSize[rate_random][days_random],
                                                (x, y, z), diameter_int, min,
                                               max)
            numpy.save(os.path.join(images_file,
                                   "{}_{}_{}_diameter_int_{}.npz".format(str(i),writefilename_str, str(number_int),
                                                                        str(diameter_int))),


                       image_data_numpy,
                          )
            mask_data_numpy = getConv.get_data(imageDataSize[rate_random][days_random+1],
                                                (x, y, z), diameter_int, min,
                                               max)
            numpy.save(os.path.join(masks_file,
                                   "{}_{}_{}_diameter_int_{}.npz".format(str(i),writefilename_str, str(number_int),
                                                                        str(diameter_int))),
                          mask_data_numpy,
                          )


            if i % 10000 == 0:
                print("{} Done.".format(i))

        return True
    else:
        return imageDataSize


def gen_compare_crop_dataset(path, save_path, number, crop_size, save_gen_log_file_name, dataset = "default", days_list= "default"):
    '''
        生成指定路径下的第t天和t+1天对应匹配的，指定图像大小和指定数量数据集
    :param path: 原数据集地址，从该地址读取并生成指定符合要求的随机数据集
    :param save_path: 生成数据保存地址
    :param number: 生成新数据的数量
    :param crop_size: 指定新生成数据剪裁大小,如（32，32）
    :param save_gen_log_file_name: 指定生成数据集的记录日志文件名
    :param dataset: 默认读取数据集，可以手动传list格式更改
    :param days_list:默认天数的list，可以手动传list格式更改
    :return:
    '''
    if dataset == "default":
        dataset = ['A35', 'A45', 'B35', 'C45']
    else:
        dataset = dataset
    if days_list == "default":
        days_list = ['3d','3d','4d','5d','6d','7d','14d','21d','28d']
    else:
        days_list = days_list

    image_dir_path = os.path.join(save_path,"images")
    mask_dir_path  = os.path.join(save_path,"masks")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(image_dir_path)
        os.mkdir(mask_dir_path)
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)
    if not os.path.exists(mask_dir_path):
        os.mkdir(mask_dir_path)

    with open(os.path.join(save_path, "{}_{}_{}.txt".format(save_gen_log_file_name, "data_info", str(time.time()))), "w") as f:
        for i in range(number):
            temp_dataset_number, temp_day_number  = random.randint(0, len(dataset) - 1), random.randint(0,
                                                                                                       len(days_list) - 1 - 1)
            temp_dataset_path_first_day = os.path.join(path, dataset[temp_dataset_number], dataset[temp_dataset_number],
                                                       days_list[temp_day_number])
            temp_dataset_path_next_day = os.path.join(path, dataset[temp_dataset_number], dataset[temp_dataset_number],
                                                      days_list[temp_day_number + 1])
            temp_file_number = random.randint(0,len(os.listdir(temp_dataset_path_first_day))-1)


            image_first_day = Image.open(os.path.join(temp_dataset_path_first_day,os.listdir(temp_dataset_path_first_day)[temp_file_number]))
            image_next_day = Image.open(os.path.join(temp_dataset_path_next_day,os.listdir(temp_dataset_path_next_day)[temp_file_number]))


            size_x, size_y = crop_size
            x, y = image_first_day.size
            if x < size_x or y < size_y:
                return "Crop size is larger than image size."
            point_x, point_y = random.randint(0, x - size_x), random.randint(0, y - size_y)

            crop_image_f, point_f = image_first_day.crop((point_x, point_y, point_x + size_x, size_y + point_y)), (point_x, point_y)
            crop_image_n, point_n = image_next_day.crop((point_x, point_y, point_x + size_x, size_y + point_y)), (point_x, point_y)

            crop_image_f.save(os.path.join(image_dir_path, "{}.bmp".format(str(i))))
            crop_image_n.save(os.path.join(mask_dir_path, "{}.bmp".format(str(i))))
            f.writelines(
                "{}:oir_image_path:{} and the next day; crop_size:{}; start_point:{};\n\r".format(str(i), temp_dataset_path_first_day, str(crop_size),
                                                                              str(point_f)))
            if i % 1000 == 0:
                print("{} done".format(str(i)))

def get_specify_series_days_crop_data(dataset_path_str,image_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720)):
    '''
    生成指定天数序列的时序2d图像数据集
    获取指定数量，天数(一段时间内,如2，3，4，5，6天)和尺寸（水泥块儿的大小，如64^2）的序列数据集
    :param dataset_path_str:
    :param image_size_int:
    :param data_number:
    :param save_path:
    :param save_file_name:
    :param dataset_list:
    :param day_list:
    :param data_size:
    :return:
    '''
    if dataset_list == 'default':
        dataset_list = ['A35', 'A45', 'B35', 'C45']
    else:
        dataset_list = dataset_list
    if day_list == 'default':
        day_list = ['2d', '3d', '4d', '5d', '6d', '7d', '14d', '21d', '28d']
    else:
        day_list = day_list

    for i in range(len(dataset_list)):
        dataset_list[i] = os.path.join(dataset_path_str, dataset_list[i], dataset_list[i])


    final_path = os.path.join(save_path, save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    with open(os.path.join(save_path, "{}_{}_{}.txt".format(save_file_name, "data_info", str(time.time()))), "w") as f:
        for num in range(data_number):

            layer_num = random.randint(0,image_size_int-1)
            start_point_x = random.randint(0,data_size[0]-image_size_int-1)
            start_point_y = random.randint(0,data_size[1]-image_size_int-1)
            crop_area = (start_point_x,start_point_y, start_point_x+image_size_int,start_point_y+image_size_int)

            gen_res_path = os.path.join(final_path,str(num))
            if not os.path.exists(gen_res_path):
                os.mkdir(gen_res_path)

            dataset_path = random.choice(dataset_list)

            for num_days in range(len(day_list)):
                temp_day_file_path = day_list[num_days]
                day_path = os.path.join(dataset_path, temp_day_file_path)

                file_list = os.listdir(day_path)
                # print(os.path.join(dataset_path, temp_day_file_path, file_list[layer_num]))
                img = Image.open(os.path.join(dataset_path, temp_day_file_path, file_list[layer_num]))
                # print(crop_area)
                img = img.crop(crop_area)
                img.save("{}/{}.bmp".format(str(gen_res_path), str(num_days)))
            f.writelines(
                "{}:oir_image_path:{}and_day_list:{}_crop_size:{}_crop_area:{}\n\r".format(str(num), dataset_path, str(day_list),str(image_size_int),
                                                                      str(crop_area)))
            if  num % int(data_number * 0.1) == 0:
                print("{}/{} done.".format(str(num), str(data_number)))
    print("All Done.")

def image_trans_bin(img_path,bin):
    '''
    将图像像素都归一到指定bin中，向前取整。如划分8个bin，0-256像素，0-32一个bin，32-64一个bin，依次类推，如40像素，在32-64bin中，将被设置为32；
    :param img_path: 图像地址
    :param bin: 指定bin个数
    :return: 返回新图像，PIL.Image
    '''
    img = Image.open(img_path)
    img = np.array(img)
    for i in range(1, bin):
        img[(img > i * (256 / bin)) & (img < (i + 1) * (256 / bin))] = i * (256 / bin)

    img = Image.fromarray(img)
    return img


if __name__ == "__main__":
    # get_data_3d(r"D:\download\work\work",['B35','C45'],['2d','3d','4d','5d','6d','7d','14d','21d','28d'],size=720,wirtetofile_bool=True,writepath_str="../data/",writefilename_str="3d",diameter_int=64,min=0,max=720,number_int=100)


    #制作t和t+1时刻匹配的数据集
    path = r"D:\download\work\Research\Datasets\ori"
    save_path = r"./"
    number = 1 * 3
    crop_size = (128,128)
    save_gen_log_file_name = "train"
    days_list = ['2d','7d','14d','21d','28d']
    dataset = ['B35', 'C45']
    gen_compare_crop_dataset(path=path,save_path=save_path,number=number,crop_size=crop_size,dataset=dataset,save_gen_log_file_name=save_gen_log_file_name,days_list=days_list)
    print("All Done")
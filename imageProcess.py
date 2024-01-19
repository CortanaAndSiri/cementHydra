import random
import time

import numpy
from PIL import Image
import numpy as np
import os
from random import  choice
import matplotlib.pyplot as plt
import imageio
import glob
import torch
import open3d.cpu.pybind.geometry


file_path = r"D:\download\work\Research\work\B35"
img_path = r"./B35_2d__rec0026.bmp"
bin = 16

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


def from_files_gen_gif(image_file_path,gif_name,fps = 1,loop = 0,image_type='png'):
    """
    读取文件夹内所有指定后缀的图像，制作成一张动图
    :param image_file_path: 输入文件夹路径
    :param gif_name: 生成gif图像名称
    :param fps: 生成图像桢数
    :param loop: gif是否循环
    :param image_type: 要求指定类型图像的后缀
    :return:
    """
    glob_path = os.path.join(image_file_path,"*.{}".format(image_type))
    files_name = glob.glob(glob_path)
    pics_list = []
    for image_name in files_name:
        im = Image.open(image_name)
        pics_list.append(im)
    imageio.mimsave(os.path.join(image_file_path,gif_name),pics_list,"GIF",fps=fps,loop=loop)


def get_filenames(file_dir,file_type):
    '''
    获取指定路径下，文件及其子文件下所有后缀名的文件地址，返回list
    :param file_dir: 指定文件路径
    :param file_type: 指定文件类型，如.bmp
    :return:返回一个包含所有符合要求的文件地址的list
    '''
    filenames=[]
    if not os.path.exists(file_dir):
        return -1
    for root,dirs,names in os.walk(file_dir):
        for filename in names:
            if os.path.splitext(filename)[1] == file_type:
                filenames.append(os.path.join(root,filename))
    return filenames


#进行图片的复制拼接
def concat_images(COL,ROW,image_names, name, path,UNIT_WIDTH_SIZE,UNIT_HEIGHT_SIZE,SAVE_QUALITY=50):
    '''
    将指定路径下的图像按照ROW行，COL列拼接起来，并返回
    :param COL: 列
    :param ROW: 行
    :param image_names:传入图像地址list
    :param name: 保存生成图像的名称
    :param path: 保存地址
    :param UNIT_WIDTH_SIZE: 单一图像宽度
    :param UNIT_HEIGHT_SIZE: 单一图像长度
    :param SAVE_QUALITY: 生成图像质量，可选0-100，默认50
    :return: 返回生成的图像，PIL.Image对象
    '''
    image_files = []
    for index in range(COL*ROW):
        image_files.append(Image.open(path + image_names[index])) #读取所有用于拼接的图片
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            #对图片进行逐行拼接
            #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            #或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(path + name + '.jpg', quality=SAVE_QUALITY) #成品图保存
    return target


def gen_lots_of_res_images(bin_list,COL,ROW,image_list,path='./',SAVE_QUALITY=50):
    '''
    对image_list中图像选：ROW行，COL列的bin_list图像（比如[4,8,16]，把256像素分别分为4，8，16个bin，在该bin的像素向前取整，并把这三张新图像和原图一起拼成新图像）进行拼接成一张大图
    :param bin_list: 分bin的list，如[4,8,16,32]
    :param COL: 拼接图像有多少列
    :param ROW: 拼接图像有多少行
    :param image_list: 需求随机采样的图像采样list，存放许多图像的地址的list
    :param path: 保存生成图像的地址
    :param SAVE_QUALITY: 生成图像质量，可选0-100，默认50
    :return: 返回生成的图像对象，PIL.Image类型
    '''
    time_str = str(time.time())
    txt_file_name = "all_image_names_{}_bin_list_{}_COL_{}_ROW_{}.txt".format(time_str,str(bin_list),str(COL),str(ROW))
    gen_image_file_name = "gen_images_names_{}_bin_list_{}_COL_{}_ROW_{}.png".format(time_str,str(bin_list),str(COL),str(ROW))
    with open(os.path.join(path,txt_file_name),'w') as f:
        image_files = []
        UNIT_WIDTH_SIZE,UNIT_HEIGHT_SIZE = Image.open(choice(image_list)).size
        for index in range(COL*ROW):
            single_one_files_list = []
            random_choice_path = choice(image_list)
            f.writelines(random_choice_path+"\r\n")
            single_one_files_list.append(Image.open(random_choice_path))
            for num in bin_list:
                single_one_files_list.append(image_trans_bin(random_choice_path, num))
            image_files.append(single_one_files_list) #读取所有用于拼接的图片
        target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL* (len(bin_list)+1), UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
        #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
        for row in range(ROW):
            for col in range(COL):
                #对图片进行逐行拼接
                #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
                #或四元元组（指定复制位置的左上角和右下角坐标）
                # target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
                for num in range(len(bin_list)+1):
                    target.paste(image_files[COL * row + col][num], (0 + UNIT_WIDTH_SIZE * ((len(bin_list)+1) * col + num), 0 + UNIT_HEIGHT_SIZE * row))
        target.save(os.path.join(path,gen_image_file_name),quality=SAVE_QUALITY)
        return target


def show_changes_in_the_same_sectin(image_path_list,is_random=True,point_x=10,point_y=10,is_many=False,ROW=1,COL=1,size=(1,1)):
    # 展示相同区域，不同时间t下的，区域像素点波动情况
    # :param image_path_list: 图像地址list
    #   如['D:\\download\\work\\Research\\work\\B35\\B35\\2d\\B35_2d__rec0182.bmp', 'D:\\download\\work\\Research\\work\\B35\\B35\\3d\\B35_3d__rec0182.bmp', ...]
    # :param is_random: 是否随机采集图像上的像素点
    # :param is_random: 是否随机采集图像上的像素点
    # :param point_x: 不随机情况下，指定像素点x的位置
    # :param point_y: 不随机情况下，指定像素点y的位置
    # :param is_many: 是否采集多次，生成多张图像拼接成一大张图像
    # :param ROW: 多次采集时，采集多少行图像
    # :param COL: 多次采集时，采集多少列图像
    # :param size: 采集图像的尺寸，默认为（1，1）一个像素点
    # :return: 返回None,直接展示plt绘图


   if is_many:
       fig,ax = plt.subplots(ROW,COL)
       plt.subplots_adjust(left=0.125,
                           bottom=0.1,
                           right=0.9,
                           top=0.9,
                           wspace=0.2,
                           hspace=0.35)
       for i in range(ROW):
           for j in range(COL):
               temp_image_list = []
               pixel_list = []
               for path in image_path_list:
                   temp_image_list.append(Image.open(path))

               if is_random:
                   x_max, y_max = temp_image_list[0].size
                   size_x, size_y = size
                   x, y = random.randint(0, x_max -size_x - 1), random.randint(0, y_max -size_y -1)
               else:
                   x, y = point_x, point_y
               for img in temp_image_list:
                   # print((x,y),size)
                   pixel_list.append(get_specify_size_avg_point(img,(x,y),size))
               ax[i][j].plot(pixel_list)
               ax[i][j].set_ylim((0, 255))
               ax[i][j].set_title("{}_x{}_y{}_s{}".format(os.path.basename(image_path_list[0]).split('.')[0], str(x), str(y),str(size)))
       plt.show()
   else:
       temp_image_list = []
       pixel_list = []
       for path in image_path_list:
           temp_image_list.append(Image.open(path))

       if is_random:
           x_max, y_max = temp_image_list[0].size
           x, y = random.randint(0, x_max - 1), random.randint(0, y_max)
       else:
           x, y = point_x, point_y
       for img in temp_image_list:
           img = np.array(img)
           pixel_list.append(img[x][y])
       plt.plot(pixel_list)
       plt.ylim((0, 255))
       plt.title("{}_x{}_y{}_s{}".format(os.path.basename(image_path_list[0]).split('.')[0], str(x), str(y),str(size)))
       plt.show()


def show_changes_in_the_same_sectin_3d(dataset_path_str,point_tuple,cube_size_int,is_point_random_bool=False,is_many=False,ROW=1,COL=1,dataset_list = 'default',day_list = 'default',data_size = (720,720,720)):
    '''
    对比相同3d区域，不同时间，点和区域内均值的误差
    :param dataset_path_str:
    :param point_tuple:
    :param cube_size_int:
    :param is_point_random_bool:
    :param is_many:
    :param ROW:
    :param COL:
    :param dataset_list:
    :param day_list:
    :param data_size:
    :return:
    '''
    if dataset_list == 'default':
        dataset_list = ['A35','A45','B35','C45']
    else:
        dataset_list = dataset_list
    if day_list == 'default':
        day_list = ['2d','3d','4d','5d','6d','7d','14d','21d','28d']
    else:
        day_list = day_list
    for i in range(len(dataset_list)):
        dataset_list[i] = os.path.join(dataset_path_str,dataset_list[i],dataset_list[i])
    dataset = np.zeros(list(data_size))
    if is_many:
        fig, ax = plt.subplots(ROW, COL)
        plt.subplots_adjust(left=0.125,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.35)
        for i in range(ROW):
            for j in range(COL):
                if is_point_random_bool:
                    d_x, d_y, d_z = data_size
                    point = (random.randint(cube_size_int // 2, (d_x - cube_size_int // 2) - 1),
                             random.randint(cube_size_int // 2, (d_y - cube_size_int // 2) - 1),
                             random.randint(cube_size_int // 2, (d_z - cube_size_int // 2) - 1))
                else:
                    point = point_tuple

                dataset_path = random.choice(dataset_list)
                res_list = []

                for i_in in range(len(day_list)):
                    temp_day_file_path = os.path.join(dataset_path, day_list[i_in])
                    file_list = os.listdir(temp_day_file_path)
                    for j_in in range(len(file_list)):
                        image_path = os.path.join(temp_day_file_path, file_list[j_in])
                        image = Image.open(image_path)
                        dataset[j_in] = np.array(image)
                    specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
                    res_list.append(specify_cube.mean())
                # print("{} days done".format(day_list[i]))
                ax[i][j].plot(res_list)
                ax[i][j].set_ylim((0, 255))
                print("{}_{} done.".format(str(i),str(i)))
                ax[i][j].set_title(
                    "path{}_point{}_size{}".format(os.path.basename(dataset_path), str(point), str(cube_size_int)))
        plt.show()

    else:
        if is_point_random_bool:
            d_x,d_y,d_z = data_size
            point = (random.randint(cube_size_int//2,(d_x-cube_size_int//2)-1), random.randint(cube_size_int//2,(d_y-cube_size_int//2)-1), random.randint(cube_size_int//2,(d_z-cube_size_int//2)-1))
        else:
            point = point_tuple

        dataset_path = random.choice(dataset_list)
        res_list = []

        for i in range(len(day_list)):
            temp_day_file_path = os.path.join(dataset_path, day_list[i])
            file_list = os.listdir(temp_day_file_path)
            for j in range(len(file_list)):
                image_path = os.path.join(temp_day_file_path, file_list[j])
                image = Image.open(image_path)
                dataset[j] = np.array(image)
            specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
            res_list.append(specify_cube.mean())
            print("{} days done".format(day_list[i]))
        plt.plot(res_list)
        plt.ylim((0, 255))
        plt.title(
            "path{}_point{}_size{}".format(os.path.basename(dataset_path), str(point), str(cube_size_int)))
        plt.show()


def show_changes_in_the_specified_layers(dataset_path_str,layers_int,is_layers_random_bool=False,is_many=False,ROW=1,COL=1,dataset_list = 'default',day_list = 'default',data_size = (720,720,720),detials=False):
    '''
    对比相同截面区域，不同时间，点和区域内均值的误差
    :param dataset_path_str:
    :param layers_int:
    :param is_layers_random_bool:
    :param is_many:
    :param ROW:
    :param COL:
    :param dataset_list:
    :param day_list:
    :param data_size:
    :param detials:
    :return:
    '''
    if dataset_list == 'default':
        dataset_list = ['A35','A45','B35','C45']
    else:
        dataset_list = dataset_list
    if day_list == 'default':
        day_list = ['2d','3d','4d','5d','6d','7d','14d','21d','28d']
    else:
        day_list = day_list
    for i in range(len(dataset_list)):
        dataset_list[i] = os.path.join(dataset_path_str,dataset_list[i],dataset_list[i])
    dataset = np.zeros(list(data_size[1:]))
    if is_many:
        fig, ax = plt.subplots(ROW, COL)
        plt.subplots_adjust(left=0.125,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.35)
        for i in range(ROW):
            for j in range(COL):
                if is_layers_random_bool:
                    layers = random.randint(0,data_size[0]-1)

                else:
                    layers = layers_int

                dataset_path = random.choice(dataset_list)
                res_list = []

                for i_in in range(len(day_list)):
                    temp_day_file_path = os.path.join(dataset_path, day_list[i_in])
                    file_list = os.listdir(temp_day_file_path)
                    image_path = os.path.join(temp_day_file_path, file_list[layers])
                    image = Image.open(image_path)
                    dataset = np.array(image)
                    res_list.append(dataset.mean())
                # print("{} days done".format(day_list[i]))
                ax[i][j].plot(res_list)
                if not detials:
                    ax[i][j].set_ylim((0, 255))
                print("{}_{} done.".format(str(i),str(i)))
                ax[i][j].set_title(
                    "path{}_layers{}_size{}".format(os.path.basename(dataset_path), str(layers), str(data_size[1:])))
        plt.show()

    else:
        if is_layers_random_bool:
            layers = random.randint(0,data_size[0]-1)
        else:
            layers = layers_int

        dataset_path = random.choice(dataset_list)
        res_list = []

        for i in range(len(day_list)):
            temp_day_file_path = os.path.join(dataset_path, day_list[i])
            file_list = os.listdir(temp_day_file_path)
            image_path = os.path.join(temp_day_file_path, file_list[layers])
            image = Image.open(image_path)
            dataset = np.array(image)
            res_list.append(dataset.mean())
            print("{} days done".format(day_list[i]))
        plt.plot(res_list)
        if not detials:
            plt.ylim((0, 255))
        plt.title(
            "path{}_point{}_size{}".format(os.path.basename(dataset_path), str(layers), str(data_size[1:])))
        plt.show()


def conv_3d(dataset,point,diameter=1,min=0,max=720):
    '''
    返回直径为diameter的三维块儿
    :param dataset: 一天的数据集
    :param point: 中心点坐标
    :param radius: 半径i
    :param min: 取值最小点
    :param max: 取值最小值
    :return: 三维取中心点指定半径的卷积取值块儿
    '''

    ori_x,ori_y,ori_z = point
    d = diameter
    data_torch_list = torch.empty([d]*3)
    select_range = range(min,max)
    if diameter % 2 != 0:
        if diameter == 1:
            radius = 1
        else:
            radius = int((diameter-1)/2)
        for z in range(ori_z - radius, ori_z + radius + 1):
            for y in range(ori_y - radius, ori_y + radius + 1):
                for x in range(ori_x - radius, ori_x + radius + 1):
                    if (x not in select_range) or (y not in select_range) or (z not in select_range):
                        data_torch_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = 0
                    else:
                        data_torch_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = dataset[z][y][
                            x]
        return data_torch_list
    else:
        radius = int(diameter/2)
        for z in range(ori_z - radius, ori_z + radius):
            for y in range(ori_y - radius, ori_y + radius):
                for x in range(ori_x - radius, ori_x + radius):
                    if (x not in select_range) or (y not in select_range) or (z not in select_range):
                        data_torch_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = 0
                    else:
                        data_torch_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = dataset[z][y][
                            x]
        return data_torch_list


def get_specified_data(dataset_t0, point, diameter=3, min=0, max=720):
    '''

    :param dataset_t0: t0时刻数据集
    :param point: 中心点坐标
    :param diameter: 直径
    :param min: 取值最小点
    :param max: 取值最小值
    :return: t0时刻三维取中心点指定半径的卷积取值块儿,直接返回numpy对象
    '''
    data = conv_3d(dataset_t0, point, diameter, min, max)
    return data


def get_specify_size_avg_point(image,point,size):
    '''
    获取指定size的区域图像内，所有像素点的平均值
    :param image: 输入图像
    :param point: 需要截取图像的，左上角点的坐标
    :param size: size，需要截取图像的尺寸，
    :return:截取指定size图像的所有像素的均值
    '''
    point_x,point_y = point
    size_x,size_y = size
    img = image.crop((point_x,point_y,point_x+size_x,size_y+point_y))
    img = np.array(img)
    return img.mean()


def get_same_place_different_days_image(file_path,specify_days_list = ['2d','3d','4d','5d','6d','7d','14d','21d','28d']):
    '''
    获取不同天数下的相同地方的水泥图像地址
    :param file_path: 文件存放地址，该地址下有对应指定文件list的各个文件夹
    :param specify_days_list: 指定获取的天数
    :return: 返回指定图像地址的list
    如['D:\\download\\work\\Research\\work\\B35\\B35\\2d\\B35_2d__rec0182.bmp', 'D:\\download\\work\\Research\\work\\B35\\B35\\3d\\B35_3d__rec0182.bmp', ...]
    '''
    image_list = []
    days_list = []
    for day in specify_days_list:
        days_list.append(os.path.join(file_path,day))
    file_len = len(os.listdir(days_list[0]))
    random_num = random.randint(0,file_len-1)
    for days_path in days_list:
        image_list.append(os.path.join(days_path,os.listdir(days_path)[random_num]))
    return image_list


def random_crop_image(image,size):
    '''
    返回指定剪裁大小的随机位置图像
    :param image: 原图
    :param size: 剪裁图像大小，如（32，32）
    :return:
    '''
    size_x,size_y = size
    x,y = image.size
    if x < size_x or y < size_y:
        return None
    point_x,point_y = random.randint(0,x -size_x),random.randint(0,y-size_y)
    return image.crop((point_x, point_y, point_x + size_x, size_y + point_y)),(point_x,point_y)


def get_crop_datasite(file_path,data_number,crop_size,save_path,save_file_name):
    '''
        生成指定大小的剪裁指定大小的图像数据集
    :param file_path:数据集地址
    :param data_number:生成数量
    :param crop_size: 剪裁大小，参数如（32,32）
    :param save_path:保存地址
    :param save_file_name:保存log文件名儿
    :return:
    '''
    image_list = get_filenames(file_path,".bmp")
    final_path = os.path.join(save_path,save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    with open(os.path.join(save_path,"{}_{}_{}.txt".format(save_file_name,"data_info",str(time.time()))),"w") as f:
        for i in range(data_number):
            temp_name = random.choice(image_list)
            image = Image.open(temp_name)
            crop_image, point = random_crop_image(image, crop_size)
            crop_image.save(os.path.join(final_path,"{}.bmp".format(str(i))))
            f.writelines("{}:oir_image_path:{}_crop_size:{}_start_point:{}\n\r".format(str(i),temp_name,str(crop_size),str(point)))
            if i%1000 == 0:
                print("{} done".format(str(i)))


def get_crop_datasite_with_bins(file_path,data_number,crop_size,save_path,save_file_name,bins):
    '''
        生成指定bins的剪裁指定大小的图像数据集
    :param file_path:数据集地址
    :param data_number:生成数量
    :param crop_size: 剪裁大小，参数如（32,32）
    :param save_path:保存地址
    :param save_file_name:保存log文件名儿
    :param bins:分bin数量
    :return:
    '''
    image_list = get_filenames(file_path,".bmp")
    final_path = os.path.join(save_path,save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    with open(os.path.join(save_path,"{}_{}_{}.txt".format(save_file_name,"data_info",str(time.time()))),"w") as f:
        for i in range(data_number):
            temp_name = random.choice(image_list)
            image = image_trans_bin(temp_name,bins)
            # image = Image.open(temp_name)
            crop_image, point = random_crop_image(image, crop_size)
            crop_image.save(os.path.join(final_path,"{}.bmp".format(str(i))))
            f.writelines("{}:oir_image_path:{}_crop_size:{}_start_point:{}\n\r".format(str(i),temp_name,str(crop_size),str(point)))
            if i%1000 == 0:
                print("{} done".format(str(i)))


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

    image_dir_path = os.path.join(save_path,"image")
    mask_dir_path  = os.path.join(save_path,"mask")
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
    :param cube_size_int:
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
                "{}:oir_image_path:{}and_day_list:{}_crop_size:{}_crop_area:{}\n\r".format(str(num), dataset_path, str(day_list),str(cube_size_int),
                                                                      str(crop_area)))
            if  num % int(data_number * 0.1) == 0:
                print("{}/{} done.".format(str(num), str(data_number)))
    print("All Done.")


def get_crop_3d_dataset(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720,720)):
    '''
    获取指定数量，指定size的3d数据集
    :param dataset_path_str:
    :param point_tuple:
    :param cube_size_int:
    :param data_number:
    :param save_path:
    :param save_file_name:
    :param dataset_list:
    :param day_list:
    :param data_size:
    :return:
    '''
    if dataset_list == 'default':
        dataset_list = ['A35','A45','B35','C45']
    else:
        dataset_list = dataset_list
    if day_list == 'default':
        day_list = ['2d','3d','4d','5d','6d','7d','14d','21d','28d']
    else:
        day_list = day_list
    for i in range(len(dataset_list)):
        dataset_list[i] = os.path.join(dataset_path_str,dataset_list[i],dataset_list[i])
    dataset = np.zeros(list(data_size))
    d_x, d_y, d_z = data_size

    final_path = os.path.join(save_path, save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    with open(os.path.join(save_path,"{}_{}_{}.txt".format(save_file_name,"data_info",str(time.time()))),"w") as f:
        for num in range(data_number):
            point = (random.randint(cube_size_int // 2, (d_x - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_y - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_z - cube_size_int // 2) - 1))

            dataset_path = random.choice(dataset_list)

            temp_day_file_path = random.choice(day_list)
            day_path = os.path.join(dataset_path,temp_day_file_path)
            file_list = os.listdir(day_path)
            for j_in in range(len(file_list)):
                image_path = os.path.join(dataset_path,temp_day_file_path, file_list[j_in])
                image = Image.open(image_path)
                dataset[j_in] = np.array(image)
            specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
            torch.save(specify_cube,"{}/{}.tensor".format(str(final_path),str(num)))
            f.writelines(
                "{}:oir_image_path:{}_crop_size:{}_start_point:{}\n\r".format(str(num), day_path, str(cube_size_int),
                                                                              str(point)))
            if num % 1000 == 0:
                print("{}/{} done.".format(str(num), str(data_number)))


def get_specify_days_crop_3d_data(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720,720)):
    '''
    生成指定天数序列的时序3dtensor数据集
    获取指定数量，天数(一段时间内,如2，3，4，5，6天)和尺寸（水泥块儿的大小，如64^3）的序列数据集
    :param dataset_path_str:
    :param cube_size_int:
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
    dataset = np.zeros(list(data_size))
    d_x, d_y, d_z = data_size

    final_path = os.path.join(save_path, save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    with open(os.path.join(save_path, "{}_{}_{}.txt".format(save_file_name, "data_info", str(time.time()))), "w") as f:
        for num in range(data_number):
            point = (random.randint(cube_size_int // 2, (d_x - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_y - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_z - cube_size_int // 2) - 1))

            gen_res_path = os.path.join(final_path,str(num))
            if not os.path.exists(gen_res_path):
                os.mkdir(gen_res_path)

            dataset_path = random.choice(dataset_list)

            for num_days in range(len(day_list)):
                temp_day_file_path = day_list[num_days]
                day_path = os.path.join(dataset_path, temp_day_file_path)
                file_list = os.listdir(day_path)
                for j_in in range(len(file_list)):
                    image_path = os.path.join(dataset_path, temp_day_file_path, file_list[j_in])
                    image = Image.open(image_path)
                    dataset[j_in] = np.array(image)
                specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
                torch.save(specify_cube, "{}/{}.tensor".format(str(gen_res_path), str(num_days)))
            f.writelines(
                "{}:oir_image_path:{}and_day_list:{}_crop_size:{}_start_point:{}\n\r".format(str(num), dataset_path, str(day_list),str(cube_size_int),
                                                                      str(point)))
            if  num % int(data_number * 0.1) == 0:
                print("{}/{} done.".format(str(num), str(data_number)))
    print("All Done.")


def get_crop_3d_dataset_and_next_day(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720,720)):
    '''
    获取指定数量，指定size的第t天和t+1天3d数据集
    :param dataset_path_str:
    :param point_tuple:
    :param cube_size_int:
    :param data_number:
    :param save_path:
    :param save_file_name:
    :param dataset_list:
    :param day_list:
    :param data_size:
    :return:
    '''
    if dataset_list == 'default':
        dataset_list = ['A35','A45','B35','C45']
    else:
        dataset_list = dataset_list
    if day_list == 'default':
        day_list = ['2d','3d','4d','5d','6d','7d','14d','21d','28d']
    else:
        day_list = day_list
    for i in range(len(dataset_list)):
        dataset_list[i] = os.path.join(dataset_path_str,dataset_list[i],dataset_list[i])
    dataset = np.zeros(list(data_size))
    d_x, d_y, d_z = data_size

    final_path = os.path.join(save_path, save_file_name)
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    first_day_path = os.path.join(final_path,"firstday")
    if not os.path.exists(first_day_path):
        os.mkdir(first_day_path)

    next_day_path = os.path.join(final_path,"nextday")
    if not os.path.exists(next_day_path):
        os.mkdir(next_day_path)

    with open(os.path.join(save_path,"{}_{}_{}.txt".format(save_file_name,"data_info",str(time.time()))),"w") as f:
        for num in range(data_number):
            point = (random.randint(cube_size_int // 2, (d_x - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_y - cube_size_int // 2) - 1),
                     random.randint(cube_size_int // 2, (d_z - cube_size_int // 2) - 1))

            dataset_path = random.choice(dataset_list)

            random_day = random.randint(0,len(day_list)-1 -1) #再减1，表示t天取值范围为[:-1],最后一天不取，若取最后一天，无下一天对应

            day_path_t = os.path.join(dataset_path,day_list[random_day])
            day_path_next = os.path.join(dataset_path,day_list[random_day+1])

            file_list_t = os.listdir(day_path_t)
            file_list_next = os.listdir(day_path_next)

            for j_in in range(len(file_list_t)):
                image_path = os.path.join(dataset_path,day_list[random_day], file_list_t[j_in])
                image = Image.open(image_path)
                dataset[j_in] = np.array(image)
            specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
            torch.save(specify_cube,"{}/{}.tensor".format(str(first_day_path),str(num)))

            for j_in in range(len(file_list_next)):
                image_path = os.path.join(dataset_path,day_list[random_day+1], file_list_next[j_in])
                image = Image.open(image_path)
                dataset[j_in] = np.array(image)
            specify_cube = conv_3d(dataset, point, cube_size_int, min=0, max=min(data_size))
            torch.save(specify_cube,"{}/{}.tensor".format(str(next_day_path),str(num)))

            f.writelines(
                "{}:oir_image_path:{} and  {} ;-crop_size:{}_start_point:{}\n\r".format(str(num), day_path_t, day_path_next, str(cube_size_int),
                                                                              str(point)))
            if num % 100 == 0:
                print("{}/{} done.".format(str(num), str(data_number)))


def show_3d_data(data):
    data = data.numpy()
    size = data.shape[0]
    xyz = np.ones((size**3,3))
    num = 0
    for k in range(1, size + 1):
        for j in range(1, size + 1):
            for i in range(1, size + 1):
                xyz[num] = [i, j, k]
                num += 1

    num = 0
    color = np.ones((size**3,3))
    for k in range(size):
        for j in range(size):
            for i in range(size):
                temp = float(data[i][j][k]) / 255.0
                color[num] = [temp, temp, temp]
                num += 1

    point_cloud = open3d.cpu.pybind.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyz)
    point_cloud.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([point_cloud])
    #在Open3D打开显示时：可以通过【Ctrl + +】来放大点。【Ctrl +--】来缩小点。鼠标滚轮来将整体放大缩小。
    #https://blog.csdn.net/hxxjxw/article/details/112382657
    #动态显示点云


def show_3d_data_res(data):
    # data = data.numpy()
    data = data.cpu().squeeze().numpy()
    size = data.shape[0]
    xyz = np.ones((size**3,3))
    num = 0
    for k in range(1, size + 1):
        for j in range(1, size + 1):
            for i in range(1, size + 1):
                xyz[num] = [i, j, k]
                num += 1

    num = 0
    color = np.ones((size**3,3))
    for k in range(size):
        for j in range(size):
            for i in range(size):
                # temp = float(data[i][j][k]) / 255.0
                temp = float(data[i][j][k])
                color[num] = [temp, temp, temp]
                num += 1

    point_cloud = open3d.cpu.pybind.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyz)
    point_cloud.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([point_cloud])
    #在Open3D打开显示时：可以通过【Ctrl + +】来放大点。【Ctrl +--】来缩小点。鼠标滚轮来将整体放大缩小。
    #https://blog.csdn.net/hxxjxw/article/details/112382657
    #动态显示点云


def data_pooling_3d(data_path,type='max'):
    data = torch.load(data_path).unsqueeze(0).unsqueeze(0)
    print(data.shape)
    m = torch.nn.MaxPool3d(3,stride=2)
    a = torch.nn.AvgPool3d(3,stride=2)
    if 'max' in type:
        output = m(data)
        for i in range(2):
            output = m(output)
    else:
        output = a(data)
        for i in range(2):
            output = a(output)
    output = output.squeeze()
    print(output.shape)
    torch.save(output,r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\pool\512\output64.tensor")


if __name__ == "__main__":

    # #展示分bin图像并拼接
    # bin_list = [4,8,16,32]
    # # for num in bin_list:
    # #     image_trans_bin(img_path, num).save("{}.bmp".format(str(num)))
    # path_list = get_filenames(file_path,'.bmp')
    # image = gen_lots_of_res_images(bin_list,COL=2,ROW=10,image_list=path_list)
    # image.show()

    # #对比相同区域，不同时间，点的误差
    # image_list = get_same_place_different_days_image(r"D:\download\work\Research\work\B35\B35")
    # print(image_list)
    # show_changes_in_the_same_sectin(image_list,is_many=True,ROW=5,COL=5,size=(13,13))

    # #对比相同3d区域，不同时间，点和区域内均值的误差
    # path = r"D:\download\work\Research\work"
    # show_changes_in_the_same_sectin_3d(path,is_many=True,is_point_random_bool=True,ROW=5,COL=5,point_tuple=(350,350,350),cube_size_int=50)

    # #对比相同截面区域，不同时间截面均值点的误差
    # path = r"D:\download\work\Research\work"
    # show_changes_in_the_specified_layers(path,layers_int=3,is_many=True,is_layers_random_bool=True,COL=5,ROW=5,dataset_list=['B35'],detials=False)

    # #随机剪裁图像测试
    # image = Image.open(r"D:\download\work\Research\work\B35\B35\14d\B35_14d__rec0028.bmp")
    # image = random_crop_image(image,(128,128))[0]
    # image.save(r"D:\download\work\Research\code\V0.0.3.pixelcnn\test3_128plus128_2.bmp")

    # # #制作指定尺寸数据集;
    # path = r"D:\download\work\Research\Datasets\ori\B35"
    # image_size = 512
    # # get_crop_datasite(path,70000,(128,128),save_path=r"D:\download\work\Research\code\V0.0.3.pixelcnn\data\cement\1",save_file_name = "train")
    # get_crop_datasite(path,7*1000,(image_size,image_size),save_path=r"D:\download\work\Research\code\dataset\ExpData\cement\512",save_file_name = "train")


    # #制作指定尺寸和指定bins1的数据集，bins：如4个bins，0-256归一化到[0,63,127,191]这个四个值内，数值区间向前取整
    # path = r"D:\download\work\Research\work\B35"
    # # get_crop_datasite(path,70000,(128,128),save_path=r"D:\download\work\Research\code\V0.0.3.pixelcnn\data\cement\1",save_file_name = "train")
    # get_crop_datasite_with_bins(path,1024 * 7,(64,64),save_path=r"D:\download\work\Research\code\V0.0.3.pixelcnn\data\cement\1",save_file_name = "train",bins=16)



    # #制作gif图像
    # path = r"D:\download\work\Research\code\V0.0.3.pixelcnn\gif\2"
    # from_files_gen_gif(path,"test1.gif")


    #制作t和t+1时刻匹配的数据集
    path = r"D:\download\work\Research\work"
    save_path = r"D:\download\work\Research\code\V0.0.4.ARpixelcnn\data\cement"
    number = 1024 * 7
    crop_size = (64,64)
    save_gen_log_file_name = "train"
    days_list = ['3d','3d','4d','5d','6d','7d']
    gen_compare_crop_dataset(path=path,save_path=save_path,number=number,crop_size=crop_size,save_gen_log_file_name=save_gen_log_file_name,days_list=days_list)D:\download\work\Code\aic\ver0.0.3_230209\aic_django\aicService\torch_models\cement_image_generation3d\dataset\6d


    # #制作3d数据集
    # dataset_path_str = r"D:\download\work\Research\Datasets\ori"
    # cube_size_int = 128
    # data_number = 50
    # # save_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\cement"
    # # save_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\pool"
    # save_path = r"D:\download\work\Code\aic\ver0.0.3_230209\aic_django\aicService\torch_models\cement_image_generation3d\dataset\6d"
    # # save_file_name = "train"
    # save_file_name = "128"
    # get_crop_3d_dataset(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720,720))


    # #制作3d，t天和t+1天配对数据集
    # dataset_path_str = r"D:\download\work\Research\Datasets\ori"
    # cube_size_int = 64
    # data_number = 1500
    # # save_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\cementAR"
    # # save_file_name = "train"
    # get_crop_3d_dataset_and_next_day(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = 'default',data_size = (720,720,720))
    #

    # ##制作3d，指定天数序列的时序数据集
    # dataset_path_str = r"D:\download\work\Research\Datasets\ori"
    # cube_size_int = 128
    # data_number = 10
    # # save_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\seriesCementData"
    # # save_file_name = "train"
    # save_path = r"D:\download\work\Code\aic\ver0.0.3_230209\aic_django\aicService\torch_models\get28days_3DData\dataset"
    # save_file_name = "fulldays"
    # get_specify_days_crop_3d_data(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = ['2d', '3d', '4d', '5d', '6d', '7d', '14d', '21d', '28d'],data_size = (720,720,720))
    #


    # # ##制作2d，指定天数序列的时序数据集
    # dataset_path_str = r"D:\download\work\Research\Datasets\ori"
    # cube_size_int = 64
    # data_number = 8000
    # save_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\seriesCementData2d"
    # save_file_name = "train"
    # get_specify_series_days_crop_data(dataset_path_str,cube_size_int,data_number,save_path,save_file_name,dataset_list = 'default',day_list = ['7d', '14d', '21d', '28d'],data_size = (720,720,720))


    # # #3d数据集可视化_数据集可视化
    # ori_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\1.tensor"
    # # data_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\pool\512\output64.tensor"
    # data = torch.load(ori_path)
    # show_3d_data(data)


    #3d数据集可视化_网络输出结果可视化
    # data_path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\res_gatedPixcelcnn.tensor"
    # # path2 = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\res\res4\pixelcnn_res_9.tensor"
    # data = torch.load(data_path)
    # show_3d_data_res(data)

    # #pooling test
    # path = r"D:\download\work\Research\code\V0.0.5.ARpixelcnn_3d\data\pool\512\0.tensor"
    # data_pooling_3d(path,type='avg')

    pass



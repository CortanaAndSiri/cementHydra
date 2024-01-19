import numpy as np

def conv_3d_v1(dataset,point,radius=1,min=0,max=720):
    '''
    最初版本，取直径2n+1的三维块儿
    :param dataset: 一天的数据集
    :param point: 中心点坐标
    :param radius: 半径i
    :param min: 取值最小点
    :param max: 取值最小值
    :return: 三维取中心点指定半径的卷积取值块儿
    '''

    ori_x,ori_y,ori_z = point
    d = 2 * radius + 1
    data_np_list = np.empty([d]*3,dtype=int)
    select_range = range(min,max)
    for z in range(ori_z-radius,ori_z+radius+1):
        for y in range(ori_y-radius,ori_y+radius+1):
            for x in range(ori_x-radius,ori_x+radius+1):
                if (x not in select_range) or (y not in select_range) or (z not in select_range):
                    data_np_list[z-(ori_z-radius)][y-(ori_y-radius)][x-(ori_x-radius)] = 0
                else:
                    data_np_list[z-(ori_z-radius)][y-(ori_y-radius)][x-(ori_x-radius)] = dataset[z][y][x]
    return  data_np_list


def conv_3d_v2(dataset,point,radius=1,min=0,max=720):
    '''
    修改版本，取直径2n的三维块儿
    :param dataset: 一天的数据集
    :param point: 中心点坐标
    :param radius: 半径i
    :param min: 取值最小点
    :param max: 取值最小值
    :return: 三维取中心点指定半径的卷积取值块儿
    '''

    ori_x,ori_y,ori_z = point
    d = 2 * radius
    data_np_list = np.empty([d]*3,dtype=int)
    select_range = range(min,max)
    for z in range(ori_z-radius,ori_z+radius):
        for y in range(ori_y-radius,ori_y+radius):
            for x in range(ori_x-radius,ori_x+radius):
                if (x not in select_range) or (y not in select_range) or (z not in select_range):
                    data_np_list[z-(ori_z-radius)][y-(ori_y-radius)][x-(ori_x-radius)] = 0
                else:
                    data_np_list[z-(ori_z-radius)][y-(ori_y-radius)][x-(ori_x-radius)] = dataset[z][y][x]
    return  data_np_list


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
    data_np_list = np.empty([d]*3,dtype=int)
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
                        data_np_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = 0
                    else:
                        data_np_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = dataset[z][y][
                            x]
        return data_np_list
    else:
        radius = int(diameter/2)
        for z in range(ori_z - radius, ori_z + radius):
            for y in range(ori_y - radius, ori_y + radius):
                for x in range(ori_x - radius, ori_x + radius):
                    if (x not in select_range) or (y not in select_range) or (z not in select_range):
                        data_np_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = 0
                    else:
                        data_np_list[z - (ori_z - radius)][y - (ori_y - radius)][x - (ori_x - radius)] = dataset[z][y][
                            x]
        return data_np_list




def get_data(dataset_t0, point, diameter=3, min=0, max=720):
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


def get_data_and_result(dataset_t0, dataset_t1, point, diameter=3, min=0, max=720):
    '''

    :param dataset_t0: t0时刻数据集
    :param dataset_t1: t1时刻数据集
    :param point: 中心点坐标
    :param radius: 半径(直径2r+1)
    :param min: 取值最小点
    :param max: 取值最小值
    :return: t0时刻三维取中心点指定半径的卷积取值块儿展平为list，并追加t1对应中心点的像素值
    '''
    data = conv_3d(dataset_t0, point, diameter, min, max)
    data = data.flatten()
    x,y,z = point
    data = list(data)
    data.append(dataset_t1[z][y][x])
    return data
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
# from glob import glob
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.fcnplusmdn import FCNs as fcnmdnFCNs
from model.fcn import FCNs
import imageio



path_train = "./0.bmp"
path_label = "./2T.bmp"
# path_train = "./simple.bmp"
# path_label = "./simple.bmp"

image = Image.open(path_train)
image_label = Image.open(path_label)


transform_x = T.Compose([T.ToTensor()])


transform_toImage = T.ToPILImage()

image_input = transform_x(image).unsqueeze(0)




# #fcnMDN
n_gaussians = 5
fcnmdnFCNsmodel = fcnmdnFCNs(n_gaussians,first_channel=1,backbone="vgg")
net_info = torch.load(r"./fcnmdn_models/601.ckpt")
state_dict = net_info["state_dict"]
fcnmdnFCNsmodel.load_state_dict(state_dict)


# #fcn
model  = FCNs(256, "vgg",first_channel=1)
net_info = torch.load(r"./models/281.ckpt")
state_dict = net_info["state_dict"]
model.load_state_dict(state_dict)

# with torch.no_grad():
#     image_output = model(image_input)
#     output = image_output.squeeze()
#     ch_number,x_point,y_point = output.shape
#     image_pred = torch.max(output,0)
#     image = transform_toImage(image_pred[1].float())
#     image_trans = transform_toImage((255-image_pred[1]).float())
#
#     # image.show()
#     #预测图像是反色的，需要255-pred修正成原色
#
#     image_ori.show()
#     image_trans.show()
#     image_label_trans.show()

def pred_fcn(number = 50,img_input = image_input, image_size = 128):

    with torch.no_grad():
        for number in range(number):
            image_output = model(img_input).squeeze()


            image_output = image_output.permute([1,2,0])

            pred = torch.multinomial(image_output.view(-1,256),num_samples=1)
            image_output = pred.reshape(image_size,image_size).int()


            pred_image = Image.fromarray(image_output.numpy()).convert("L")
            pred_image.save(r"./res/{}.bmp".format(str(number)))

            img_input = transform_x(pred_image).unsqueeze(0)

        #     output = image_output.squeeze()
        #
        #     # image_pred = torch.max(output, 0)
        #     # image_trans = transform_toImage((255 - image_pred[1]).float())
        #
        #     image_pred = output
        #     image_trans = transform_toImage(image_pred.float())
        #     image_trans.save("{}.jpg".format(str(number)))
        #     img_input = transform_x(image_trans).unsqueeze(0)
        # image_ori.save("ori.jpg")




def pred_fcnmdn(number = 50,img_input = image_input,n_gaussians=5,image_size = 128):

    with torch.no_grad():
        for number in range(1,number+1):
            img_input


            pi, mu, sigma = fcnmdnFCNsmodel(img_input)

            # k = torch.multinomial(pi, 1).view(-1)


            pi = pi.squeeze()
            pi = pi.reshape(-1,n_gaussians)

            mu = mu.squeeze()
            mu = mu.reshape(-1, n_gaussians)

            sigma = sigma.squeeze()
            sigma = sigma.reshape(-1, n_gaussians)

            k = torch.multinomial(pi, 1).view(-1)
            # print(sigma)
            y_pred = torch.normal(mu, torch.abs(sigma))[np.arange(image_size**2), k].data


            #这里添加软约束，生成不规则的像素，重新生成
            shhape_number = sigma.shape[0]
            for num in range(shhape_number):
                temp_i = 0
                while y_pred[num].data > 255 or y_pred[num].data < 0:
                    print(y_pred[num].data)
                    y_pred[num] = torch.normal(mu[num], torch.abs(sigma[num]))[k[num]].data
                    temp_i += 1
                    if temp_i == 10:
                        break
            #.permute([1,0])
            y_pred = y_pred.reshape(image_size,image_size)
            y_pred = y_pred.int().numpy()
            pred_image = Image.fromarray(y_pred).convert('L')
            pred_image.save(r"./res/{}.bmp".format(str(number)))

            img_input = transform_x(pred_image).unsqueeze(0)

        #     output = image_output.squeeze()
        #
        #     # image_pred = torch.max(output, 0)
        #     # image_trans = transform_toImage((255 - image_pred[1]).float())
        #
        #     image_pred = output
        #     image_trans = transform_toImage(image_pred.float())
        #     image_trans.save("{}.jpg".format(str(number)))
        #     img_input = transform_x(image_trans).unsqueeze(0)
        # image_ori.save("ori.jpg")


def create_gif(path, gif_name, duration = 1.0):
    '''
    可修改变量
    :1. image_list: 这个列表用于存放生成动图的图片
    :2. gif_name: 字符串，所生成gif文件名，带.gif后缀
    :3. duration: 图像间隔时间
    :4. 在IDLE 运行的时候，将 .py 文件保存在图片存在的文件夹中
    '''
    frames = []
    image_list = os.listdir(path)
    image_list.sort(key=lambda x:int(x.split('.')[0]))
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(path,image_name)))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return




if __name__ == "__main__":
    # pred_fcnmdn(number=50,img_input=image_input)
    pred_fcn(number=50, img_input=image_input)
    # create_gif(r"./res",'compare.gif',duration=0.5)







# image_output = net(image_input)
# print(image_output)
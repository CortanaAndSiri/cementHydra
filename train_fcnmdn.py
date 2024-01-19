# encoding: utf-8
import os
import time
import numpy as np
import torch
import torch.nn as nn
import logging
from optparse import OptionParser
from model import fcnplusmdn
from datasetfcnmdn import CustomDataset
from torch.utils.data import random_split, DataLoader
from PIL import Image

CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)


#第二维和第五维度互换（1，4）
dim_change = [0,4,2,3,1]
delta = 1e-20

# def mdn_loss_fn(y, mu, sigma, pi):
#
#     delta = 1e-40
#     #处理维度不匹配，造成生成分布报错
#     #ori[patch,num_mu/pi/sigma,w,h]
#     #change [patch,1,w,h,num_mu/pi/sigma]
#
#     # mu = mu.unsqueeze(-1)
#     # mu = mu.permute(dim_change)
#     #
#     # sigma = sigma.unsqueeze(-1)
#     # sigma = sigma.permute(dim_change)
#     #
#     # pi = pi.unsqueeze(-1)
#     # pi = pi.permute(dim_change)
#
#     m = torch.distributions.Normal(loc=mu, scale=sigma)
#     loss = torch.exp(m.log_prob(y))
#     loss = torch.sum(loss * pi, dim=1)
#     #存在0，加扰动delta避免log0无穷大问题
#     loss = -torch.log(loss+delta)
#
#     return torch.mean(loss)


def mdn_loss_fn(y, mu, sigma, pi):
    # 从[32, 128, 128]，扩展为[32, 1, 128, 128, 1]
    y = y.unsqueeze(1).unsqueeze(-1)

    m = torch.distributions.Normal(loc=mu, scale=sigma)
    y = y.expand_as(sigma)
    log_prob = m.log_prob(y)
    weighted_logprob = torch.log(pi).expand_as(sigma) + log_prob
    return -torch.logsumexp(weighted_logprob, dim=1).mean()



def train(**kwargs):
    mymodel = kwargs["mymodel"]
    criterion = kwargs["criterion"]
    data_loader = kwargs["data_loader"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    save_freq = kwargs["save_freq"]
    save_dir = kwargs["save_dir"]
    verbose = kwargs["verbose"]

    start_time = time.time()
    logging.info("Epoch %03d, Learning Rate %g" % (epoch + 1, optimizer.param_groups[0]["lr"]))
    mymodel.train()

    epoch_loss = 0.0
    batches = 0
    for i, sample in enumerate(data_loader):
        image, target = sample
        if CUDA:
            image = image.to(device)
            target = target.to(device)

        optimizer.zero_grad()
        pi, mu, sigma = mymodel(image)
        # print("Sigma is +++++++++++++++++++++",sigma)
        # loss = criterion(output, target).mean()
        loss = mdn_loss_fn(target, mu, sigma, pi)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches += 1

        if (i + 1) % verbose == 0:
            logging.info('Training Loss: %.6f' % (epoch_loss / batches))

    # save checkpoint model
    if epoch % save_freq == 0:
        # 多卡训练代码
        # state_dict = mymodel.module.state_dict()
        # 单卡训练
        state_dict = mymodel.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict, },
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    end_time = time.time()
    logging.info('Batch Loss: %.6f Time: %d s' % (epoch_loss / batches, end_time - start_time))
    return epoch_loss / batches


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    mymodel = kwargs['mymodel']
    criterion = kwargs['criterion']
    verbose = kwargs['verbose']

    start_time = time.time()
    mymodel.eval()

    epoch_loss = 0.0
    for i, sample in enumerate(data_loader):
        image, target = sample
        if CUDA:
            image = image.to(device)
            target = target.to(device)
        with torch.no_grad():
            # # ori code
            # output = mymodel(image)
            # loss = criterion(output, target)

            pi, mu, sigma = mymodel(image)
            loss = mdn_loss_fn(target, mu, sigma, pi)

        epoch_loss += loss.item()

        if (i + 1) % verbose == 0:
            logging.info('Loss: %.6f' % epoch_loss)

    end_time = time.time()
    logging.info('Loss: %.6f Time: %d' % (epoch_loss, end_time - start_time))
    return epoch_loss


def test(**kwargs):
    data_loader = kwargs['data_loader']
    mymodel = kwargs['mymodel']

    start_time = time.time()
    mymodel.eval()

    for i, sample in enumerate(data_loader):
        image, path = sample
        if CUDA:
            image = image.to(device)

        with torch.no_grad():
            # output = mymodel(image)

            pi, mu, sigma = mymodel(image)
            # loss = mdn_loss_fn(target, mu, sigma, pi)

        # pred = output.data.cpu().numpy()
        # pred = np.argmin(pred, axis=1)
        # for j, p in enumerate(path):
        #     im = Image.fromarray(pred.astype('uint8')[j] * 255, "L")
        #     im.save(os.path.join("data/testPreds", p.split("\\")[-1]))

    end_time = time.time()
    logging.info('Testing Time: %d s' % (end_time - start_time))

def main():
    parser = OptionParser()
    parser.add_option("-j", "--workers", dest="workers", default=1, type="int",
                      help="number of data loading workers (default: 1)")
    parser.add_option("-e", "--epochs", dest="epochs", default=2000, type="int",
                      help="number of epochs (default: 20)")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=128, type="int",
                      help="batch size (default: 8)")
    parser.add_option("-c", "--ckpt", dest="ckpt", default=False,
                      help="load checkpoint model (default: False)")
    parser.add_option("-v", "--verbose", dest="verbose", default=100, type="int",
                      help="show information for each <verbose> iterations (default: 100)")
    parser.add_option("-n", "--num-classes", dest="num_classes", default=5, type="int",
                      help="Guassian number of classes (default: 8)")
    parser.add_option("-d", "--back-bone", dest="back_bone", default="vgg",
                      help="backbone net (default: vgg)")
    parser.add_option("-m", "--mode", dest="mode", default="train",
                      help="running mode (default: train)")

    parser.add_option("--lr", "--learn-rate", dest="lr", default=1e-3, type="float",
                      help="learning rate (default: 1e-2)")
    parser.add_option("--sf", "--save-freq", dest="save_freq", default=100, type="int",
                      help="saving frequency of .ckpt models (default: 1)")
    parser.add_option("--sd", "--save-dir", dest="save_dir", default="./fcnmdn_models",
                      help="saving directory of .ckpt models (default: ./models)")
    parser.add_option("--init", "--initial-training", dest="initial_training", default=1, type="int",
                      help="train from 1-beginning or 0-resume training (default: 1)")

    (options, args) = parser.parse_args()
    assert options.mode in ["train", "test"]
    start_epoch = 0
    mymodel = fcnplusmdn.FCNs(options.num_classes, backbone=options.back_bone,
                              first_channel=1)

    # checkpoint
    if options.ckpt:
        ckpt = options.ckpt

        if options.initial_training == 0:
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]

        mymodel.load_state_dict(state_dict)
        logging.info(f"Model loaded from {options.ckpt}")

    # initialize model-saving directory
    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # CUDA
    if CUDA:
        mymodel.to(device)
        # 并行，先注释掉
        # mymodel = nn.DataParallel(mymodel)

    # dataset
    custom_dataset = CustomDataset("data/")
    test_set = CustomDataset("data/", mode="test")

    train_size = int(0.9 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_set, val_set = random_split(custom_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=options.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=options.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=options.batch_size, shuffle=False)

    if options.mode == "test":
        test(mymodel=mymodel,
             data_loader=test_loader)
        return

    # optimizer = torch.optim.SGD(mymodel.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=options.lr)
    # optimizer = torch.optim.Adam(mymodel.parameters(), lr=options.lr)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    criterion = None
    # 这里没用MSE，自定义最大似然估计loss

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_set), len(val_set)))

    train_loss_list = []
    validate_loss_list = []
    for epoch in range(start_epoch, options.epochs):
        train_loss = train(epoch=epoch,
              data_loader=train_loader,
              mymodel=mymodel,
              criterion=criterion,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose)
        train_loss_list.append(train_loss)

        validate_loss = validate(data_loader=val_loader,
                mymodel=mymodel,
                criterion=criterion,
                verbose=options.verbose)
        validate_loss_list.append(validate_loss)
        # scheduler.step()
        torch.save(torch.tensor(train_loss_list), os.path.join(save_dir, 'train_loss_list.pth'))
        torch.save(torch.tensor(validate_loss_list), os.path.join(save_dir, 'validate_loss_list.pth'))



if __name__ == '__main__':
    main()


# encoding: utf-8


import os
import time
import numpy as np
import torch
import torch.nn as nn
import logging
from optparse import OptionParser
from model import fcn
from dataset import CustomDataset
from torch.utils.data import random_split, DataLoader
# import visdom
from PIL import Image

CUDA = torch.cuda.is_available()
logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
# vis = visdom.Visdom()


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
            image = image.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = mymodel(image)
        # print(output.shape)
        # print(target.shape)
        loss = criterion(output, target)
        # print("Loss:{}",loss)


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()		
        batches += 1


        if (i + 1) % verbose == 0:
            logging.info('Training Loss: %.12f' % (epoch_loss / batches))
            logging.info('')

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = mymodel.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    end_time = time.time()
    logging.info('Done! Train Batch Loss: %.6f Time: %d s' % (epoch_loss / batches, end_time - start_time))



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
            image = image.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = mymodel(image)

            loss = criterion(output, target)

        # pred = output.data.cpu().numpy()
        # pred = np.argmin(pred, axis=1)
        # t = np.argmin(target.cpu().numpy(), axis=1)

        # print("pred:",pred,"; t:",t)
        # vis.close()
        # vis.images(pred[:, None, :, :], opts=dict(title='pred'))
        # vis.images(t[:, None, :, :], opts=dict(title='target'))

        epoch_loss += loss.item()

        if (i + 1) % verbose == 0:
            logging.info('test Loss: %.6f' % epoch_loss)

    end_time = time.time()
    logging.info('Test Loss: %.6f Time: %d' % (epoch_loss, end_time - start_time))


def test(**kwargs):
    data_loader = kwargs['data_loader']
    mymodel = kwargs['mymodel']

    start_time = time.time()
    mymodel.eval()

    for i, sample in enumerate(data_loader):
        image, path = sample
        if CUDA:
            image = image.cuda()

        with torch.no_grad():
            output = mymodel(image)
        
        pred = output.data.cpu().numpy()
        pred = np.argmin(pred, axis=1)
        for j, p in enumerate(path):
            im = Image.fromarray(pred.astype('uint8')[j]*255, "L")
            im.save(os.path.join("data/testPreds", p.split("\\")[-1]))

    end_time = time.time()
    logging.info('Testing Time: %d s' % (end_time - start_time))


def main():
    parser = OptionParser()
    parser.add_option("-j", "--workers", dest="workers", default=1, type="int",
                      help="number of data loading workers (default: 1)")
    parser.add_option("-e", "--epochs", dest="epochs", default=300, type="int",
                      help="number of epochs (default: 50)")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=128, type="int",
                      help="batch size (default: 8)")
    parser.add_option("-c", "--ckpt", dest="ckpt", default=False,
                      help="load checkpoint model (default: False)")
    parser.add_option("-v", "--verbose", dest="verbose", default=100, type="int",
                      help="show information for each <verbose> iterations (default: 100)")
    parser.add_option("-n", "--num-classes", dest="num_classes", default=256, type="int",
                      help="number of classes (default: 1)")
    parser.add_option("-d", "--back-bone", dest="back_bone", default="vgg",
                      help="backbone net (default: vgg)")
    parser.add_option("-m", "--mode", dest="mode", default="train",
                      help="running mode (default: train)")

    parser.add_option("--lr", "--learn-rate", dest="lr", default=1e-3, type="float",
                      help="learning rate (default: 1e-3)")
    parser.add_option("--sf", "--save-freq", dest="save_freq", default=20, type="int",
                      help="saving frequency of .ckpt models (default: 1)")
    parser.add_option("--sd", "--save-dir", dest="save_dir", default="./models",
                      help="saving directory of .ckpt models (default: ./models)")
    parser.add_option("--init", "--initial-training", dest="initial_training", default=1, type="int",
                      help="train from 1-beginning or 0-resume training (default: 1)")

    (options, args) = parser.parse_args()
    assert options.mode in ["train", "test"]
    start_epoch = 0
    mymodel = fcn.FCNs(options.num_classes, options.back_bone,first_channel=1)

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
        mymodel.to(torch.device("cuda"))
        mymodel = nn.DataParallel(mymodel)

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
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_set), len(val_set)))

    for epoch in range(start_epoch, options.epochs):
        train(  epoch=epoch,
                data_loader=train_loader, 
                mymodel=mymodel,
                criterion=criterion,
                optimizer=optimizer,
                save_freq=options.save_freq,
                save_dir=options.save_dir,
                verbose=options.verbose)

        validate(data_loader=val_loader,
                mymodel=mymodel,
                criterion=criterion,
                verbose=options.verbose)
        # scheduler.step()


if __name__ == '__main__':

    main()

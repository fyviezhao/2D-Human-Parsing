import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('../')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2


# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import torch.nn.functional as F

import time

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img_list, opts, output_path='./', output_name='f', use_gpu=True):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    start_time = timeit.default_timer()
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    # adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2, 3)
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    # adj3 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7)
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    # adj2 = torch.from_numpy(graph.cihp2pascal_adj).float()
    # adj2 = adj2.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20)
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    # adj1 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20)
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    showFreq = 200
    i = 1
    sstime = time.time()
    total = len(img_list)
    for img_path in img_list:
        single_ss = time.time()
        # first make sure the result not exist
        img_path = img_path.strip('\n')
        img_split = img_path.split('/')
        img_id = img_split[-1][:-4]
        video_id = img_split[-2]
        vis_output_dir = os.path.join(opts.output_dir, 'train' + '_parsing_vis', video_id)
        output_path = vis_output_dir
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)
        output_name = '{}_vis.png'.format(img_id)
        if osp.exists(osp.join(vis_output_dir, output_name)):
            print('skip', img_path)
            continue
        if i % showFreq == 0:
            exp_time = time.time() - sstime
            print('{}/{} Finish, total time:{}'.format(str(i), str(total), str(exp_time)))
        i = i + 1
        img = read_img(osp.join(opts.data_root, img_path))
        single_ss2 = time.time()
        testloader_list = []
        testloader_flip_list = []
        for pv in scale_list:
            single_ss3 = time.time()
            composed_transforms_ts = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])
            single_ss4 = time.time()
            testloader_list.append(img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
            single_ss5 = time.time()
            print(single_ss5 - single_ss4, single_ss4 - single_ss3)
        # print(testloader_list)
        # Main Training and Testing Loop
        single_ee2 = time.time()
        print("preprocessing take time:", single_ee2 - single_ss2)
        for epoch in range(1):
            # start_time = timeit.default_timer()
            # One testing epoch
            net.eval()
            # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

            for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
                inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
                inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
                inputs = inputs.unsqueeze(0)
                inputs_f = inputs_f.unsqueeze(0)
                inputs = torch.cat((inputs, inputs_f), dim=0)
                # print('######### inputs.size() #########')
                # print(inputs.size())
                # print('#################################')
                if iii == 0:
                    _, _, h, w = inputs.size()
                # assert inputs.size() == inputs_f.size()

                # Forward pass of the mini-batch
                inputs = Variable(inputs, requires_grad=False)

                with torch.no_grad():
                    if use_gpu >= 0:
                        inputs = inputs.cuda()
                    # outputs = net.forward(inputs)
                    outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                    outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                    outputs = outputs.unsqueeze(0)

                    if iii > 0:
                        outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                        outputs_final = outputs_final + outputs
                    else:
                        outputs_final = outputs.clone()
            ################ plot pic
            predictions = torch.max(outputs_final, 1)[1]
            results = predictions.cpu().numpy()
            vis_res = decode_labels(results)

            parsing_im = Image.fromarray(vis_res[0])
            parsing_im.save(output_path+'/{}'.format(output_name))
            # cv2.imwrite(output_path+'/{}_gray.png'.format(output_name), results[0, :, :])

            end_time = timeit.default_timer()
            # print('time use for image' + ' is :' + str(end_time - start_time))
        single_ee = time.time()
        print('total time for single image ', single_ee - single_ss, single_ee - single_ss2)

if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--img_list', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    # parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--data_root', default='', type=str)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_v3v5_more_savemem(n_classes=20, os=16,
                                                                                   hidden_layers=128,
                                                                                   source_classes=7,)

    if not opts.loadmodel == '':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:', opts.loadmodel)
    else:
        print('no model load !!!!!!!!')
        raise RuntimeError('No model!!!!')

    if opts.use_gpu >0 :
        net.cuda()
        use_gpu = True
    else:
        use_gpu = False
        raise RuntimeError('must use gpu!!!!')
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    # load img list
    import os.path as osp
    file = open(osp.join(opts.data_root, opts.img_list))
    imgs = file.readlines()
    # total = len(imgs)
    # import time
    # sstime = time.time()
    # i = 1
    # showFreq = 200
    # for img in imgs:
    #     single_ss = time.time()
    #     img = img.strip('\n')
    #     img_split = img.split('/')
    #     img_id = img_split[-1][:-4]
    #     video_id = img_split[-2]
    #     vis_output_dir = os.path.join(opts.output_dir, 'train' + '_parsing_vis', video_id)
    #     if not os.path.exists(vis_output_dir):
    #         os.makedirs(vis_output_dir)
    #     output_name = '{}_vis.png'.format(img_id)
    #     if not osp.exists(osp.join(vis_output_dir, output_name)):
    #         inference(net=net, img_path=osp.join(opts.data_root, img),output_path=vis_output_dir , output_name=output_name, use_gpu=use_gpu)
    #     if i % showFreq == 0:
    #         exp_time = time.time() - sstime
    #         print('{}/{} Finish, total time:{}'.format(str(i), str(total), str(exp_time)))
    #     single_ee = time.time()
    #     print('total time for single image ', single_ee - single_ss)
    #     i = i + 1
    inference(net=net, img_list=imgs, opts=opts)

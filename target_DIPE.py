import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from lib import newSGD
# export OMP_NUM_THREADS = 1
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75,tau= 0):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['weight_decay1'] = 3e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
        param_group['tau'] = tau

    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    bestacc = 0
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC1.load_state_dict(torch.load(modelpath))
    netC.eval()
    
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC1.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    
    # newSGD includes the exploration of domain-invariant parameters
    optimizer = newSGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    sigam = 0
    while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = iter_test.next()
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = iter_test.next()

            if inputs_test.size(0) == 1:
                continue

            if  iter_num % interval_iter == 0 and args.cls_par > 0:
                netF.eval()
                netB.eval()
                netC1.eval()
                mem_label = obtain_label(dset_loaders['test'], netF, netB, netC1, args)
                mem_label = torch.from_numpy(mem_label).cuda()
                netF.train()
                netB.train()
                netC1.train()

            inputs_test = inputs_test.cuda()
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            outputs_test1 = netC1(features_test)
           

            if args.cls_par > 0:
                pred = mem_label[tar_idx]
                classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
                classifier_loss *= args.cls_par
                if iter_num < interval_iter and args.dset == "VISDA-C":
                    classifier_loss *= 0
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))

                softmax_out1 = nn.Softmax(dim=1)(outputs_test1)
                entropy_loss1 = torch.mean(loss.Entropy(softmax_out1))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss

                    msoftmax1 = softmax_out1.mean(dim=0)
                    gentropy_loss1 = torch.sum(-msoftmax1 * torch.log(msoftmax1 + args.epsilon))
                    entropy_loss1 -= gentropy_loss1
                im_loss = entropy_loss * args.ent_par
                im_loss1 = entropy_loss1 * args.ent_par
                classifier_loss += im_loss
                classifier_loss += (0.3 * im_loss1)

            iter_num += 1
            sigam  = 0.6 * (1.0 - np.float(2.0 / (1.0 + np.exp(-10 * iter_num / float(max_iter))) - 1.0))
            lr_scheduler(optimizer, iter_num = iter_num, max_iter=max_iter, tau= sigam)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
        
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                if args.dset=='VISDA-C':
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str+'\n')
                if(bestacc <= acc_s_te ):
                    bestacc = acc_s_te
                    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
                    log_s= 'best_acc: {}%'.format(acc_s_te)
                    print(log_s)
                netF.train()
                netB.train()
            
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_f1 = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_f1 = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()


    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

     #initial
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    
    all_f1 = all_f1.float().cpu().numpy()
    M = 4
    all_fliter = []
    all_filter_label = []
    num_count = 0
    #filter
    dd_self = cdist(all_f1, all_f1, args.distance)

    for i in range(len(dd_self)):
        num_preds = []
        max_ll = 0
        max_laa =0
        loc_score = []
        dic_score = {}
        num_count =  num_count + 1
        for index in range(M):
            dd_self[i][np.argmin(dd_self[i])] = 100
            num_preds.append(pred_label[np.argmin(dd_self[i])])
            loc_score.append(all_f1[np.argmin(dd_self[i])]) 

        result = dict(Counter(num_preds))
        for key in result:
            if(result[key] > max_ll):
                max_ll = result[key]
                max_laa = key  
        for index in range(M):
            if num_preds[index] in dic_score:
                dic_score[num_preds[index]]= dic_score[num_preds[index]] + loc_score[index]
            else:
                dic_score[num_preds[index]] = loc_score[index]
        if(len(result) == M):
            all_filter_label.append(pred_label[i])
            all_fliter.append(True)
        elif(len(result) == (M/2) and max_ll == 2):
            sim = -1
            replac_key = 0
            for key in dic_score:
                sim_score = cosine_similarity(all_f1[i].reshape(1,-1),(dic_score[key]/2).reshape(1,-1))[0][0].tolist()
                if sim < sim_score:
                    sim = sim_score
                    replac_key = key
            all_fliter.append(False)
            all_filter_label.append(replac_key)
        else:
            all_fliter.append(False)
            all_filter_label.append(max_laa)
    #replace 
    for i in range(len(all_fliter)):
        if(all_fliter[i] == False):
            pred_label[i] = all_filter_label[i]

    acc1 = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100,acc1 *100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DIPE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--max_epoch', type=int, default = 15, help="max iterations")
    parser.add_argument('--interval', type=int, default = 15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='./object/meckps/R_DIPE')
    parser.add_argument('--output_src', type=str, default='./object/meckps')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_world']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(3)
   

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder =  './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        # the path of source model
        model_str = names[args.s][0].lower() + '2'+ names[args.t][0].lower()
        args.output_dir_src = osp.join(args.output_src, model_str,'source', args.da, args.dset)
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
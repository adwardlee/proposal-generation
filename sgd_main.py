import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

best_prec1 = torch.Tensor([0])
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
loss_epoch = []
decay_count = 0
lowest_loss = 1000
epoch_count = 4



def main():
    global args, best_prec1
    args = parser.parse_args()
    
    ##llj 
    global loss_epoch, decay_count,lowest_loss
    global lr
    lr = args.lr

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'activitynet1.2':
        num_class = 2
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()
### llj

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            #print('checkpoint L: ', checkpoint)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.base_model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' "
                  .format(args.resume)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
#### llj

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    print(' batch size : ', args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        #### llj
        #criterion = torch.nn.CrossEntropyLoss().cpu()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(policies, args.lr,weight_decay=args.weight_decay )
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1.cpu() > best_prec1
            best_prec1 = torch.max(prec1.cpu(), best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    plt.plot(range(1,len(loss_epoch)+1),loss_epoch)
    plt.xlabel(' epoch ')
    plt.ylabel(' loss ')
    fig = plt.gcf()
    fig.savefig('validation_loss.jpg')
    plt.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
#llj

    average_loss = 0
    if args.no_partialbn:
        #llj
        model.module.partialBN(False)
        #model.partialBN(False)
    else:
        #llj
        #model.partialBN(True)
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    loss_mini_batch = 0
    batch_prec1 = 0
    batch_prec5 = 0

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #### llj
        target = target.view(-1)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)/args.iter_size
        loss_mini_batch += loss.data[0]

        # measure accuracy and record loss
        ### llj
        prec1 = accuracy(output.data, target, topk=(1,))
        batch_prec1 += prec1[0]


        # compute gradient and do SGD step
        #optimizer.zero_grad()

        loss.backward()
        
        if (i+1) % args.iter_size == 0:

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            
            losses.update(loss_mini_batch, input.size(0))
            top1.update(batch_prec1/args.iter_size, input.size(0))
            loss_mini_batch = 0
            batch_prec1 = 0

            optimizer.step()

            optimizer.zero_grad()
        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        #print('{}'.format(epoch))
        #print('{}'.format(i))
        #print('{}'.format(len(train_loader)))
        #print('{0:.3f}'.format(batch_time.avg))
        #print('{0:.3f}'.format(data_time.avg))
        #print('{0:.3f}'.format(losses.avg))
        #print('{0:.3f}'.format(top1.avg.numpy()[0]))
        #print('{0:.3f}'.format(optimizer.param_groups[-1]['lr']))
            if (i+1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}], lr: {lr: }\t'
                    'Time {batch_time.val: } ({batch_time.avg: })\t'
                    'Data {data_time.val: } ({data_time.avg: })\t'
                    'Loss {loss.val: } ({loss.avg: })\t'
                    'Prec@1 {top1_val:} ({top1_avg:})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1_val=top1.val.cpu().numpy()[0], top1_avg = top1.avg.cpu().numpy()[0],
                    lr=optimizer.param_groups[-1]['lr']))

        sys.stdout.flush()
        ### llj
        average_loss += losses.val
    average_loss /= i+1
    print('llj Training Epoch {} : average loss: {}'.format(epoch, average_loss),flush=True)
        ###
        # prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        #
        #
        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        #
        # loss.backward()
        #
        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        #
        # optimizer.step()
        #
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        #
        # if i % args.print_freq == 0:
        #     print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #            epoch, i, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    average_loss = 0
    global loss_epoch, decay_count,lowest_loss
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #### llj
        target = target.view(-1)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss

        ### llj
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val: } ({batch_time.avg: })\t'
                   'Loss {loss.val: } ({loss.avg: })\t'
                   'Prec@1 {top1_val: } ({top1_avg: })'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1_val=top1.val.cpu().numpy()[0],top1_avg=top1.avg.cpu().numpy()[0])))
        average_loss += losses.val
    if average_loss > lowest_loss:
        decay_count += 1
    else:
        lowest_loss = average_loss
        decay_count = 0
 
    loss_epoch.append(average_loss)
    print(('Testing Results: Prec@1 {top1_avg: } Loss {loss.avg: }'
        .format(top1_avg=top1.avg.cpu().numpy()[0], loss=losses)))

        #prec1, prec5 = accuracy(output.data, target, topk=(1,5))

    #     losses.update(loss.data[0], input.size(0))
    #     top1.update(prec1[0], input.size(0))
    #     top5.update(prec5[0], input.size(0))
    #
    #     # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #
    #     if i % args.print_freq == 0:
    #         print(('Test: [{0}/{1}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                i, len(val_loader), batch_time=batch_time, loss=losses,
    #                top1=top1, top5=top5)))
    #
    # print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
    #       .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best.cpu().numpy()[0]:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    ##### llj
    global lr
    global decay_count

    if decay_count > epoch_count:
        decay = 0.1
        decay_count = 0
    else:
        decay = 1
    lr = lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']



    #decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    #lr = args.lr * decay
    #decay = args.weight_decay
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr * param_group['lr_mult']
    #    param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

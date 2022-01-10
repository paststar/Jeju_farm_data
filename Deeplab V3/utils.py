import time
import torch
import logging
import os

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        #datefmt='%Y%m%d-%H:%M:%S',
                        datefmt='%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()

# def get_train_info():
#     lr = 0.01
#     l2_decay = 5e-4
#     momentum = 0.9
#     nesterov = False
#     return lr, l2_decay, momentum, nesterov

def get_data_info():
    resnet_type = 50
    num_classes = 31
    return num_classes, resnet_type

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer

def evaluate(transformer, loader,epoch,best_acc, save_dir, _print, writer):
    #_print("evaluating...")
    start = time.time()

    total = 0
    correct = 0
    transformer.eval()

    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            tgt_preds = transformer(tgt_imgs)
            pred = tgt_preds.argmax(dim=1, keepdim=True)

            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)

    eval_acc = (correct / total) * 100
    _print('(eval) target accuracy: {:.2f}% eval time: {:.2f}s'.format(eval_acc, time.time() - start))

    # if eval_acc > best_acc:
    #     best_acc = eval_acc
    # _print('(eval) test accuracy: {:.2f}% eval time: {:.2f}s'.format(eval_acc,time.time() - start))
    # _print(' best accuracy: {:.2f} % '.format(best_acc))
    # _print(' best accuracy: {:.2f} % '.format(best_acc))

    if eval_acc > best_acc:
        best_acc = eval_acc
        _print("model is saved at {} epochs".format(epoch))
        torch.save(transformer.state_dict(), save_dir + '/' + 'best_transformer.pt')

    _print('best accuracy: {:.2f} % '.format(best_acc))
    writer.add_scalar('eval/target acc', eval_acc,epoch)

    return best_acc


def get_pseudo_label(model, data):
    model.eval()
    with torch.no_grad():
        tgt_preds = model(data)
        pred = tgt_preds.argmax(dim=1)
        return pred


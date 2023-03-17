import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
#from preprocess import mean, std, preprocess_input_function
from preprocess_Mix import mean, std, preprocess_input_function

def delete_previous_models(model_path_to_delete):
    # Check if the file exists
    if os.path.exists(model_path_to_delete):
        # Delete the file
        os.remove(model_path_to_delete)
        print(f"{model_path_to_delete} deleted successfully.")
    else:
        print(f"{model_path_to_delete} does not exist.")
 
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # -gpuid=0,1,2,3
    parser.add_argument('-base_architecture', nargs=1, type=str, default='resnet50') 
    parser.add_argument('-pps_per_class', nargs=1, type=int, default='10') 
    parser.add_argument('-num_classes', nargs=1, type=int, default='6')
    parser.add_argument('-experiment_run', nargs=1, type=str, default='000')
    parser.add_argument('-run', nargs=1, type=str, default='run00')
    # python3 mainSection.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=10 -num_classes=6 -experiment_run=001 -run=run00

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0] # [0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    # book keeping namings and code
    from settingsMix import img_size, prototype_activation_function, add_on_layers_type 
    base_architecture = args.base_architecture[0]
    pps_per_class= args.pps_per_class[0]
    num_classes= args.num_classes[0]
    prototype_shape = (num_classes*pps_per_class, 128, 1, 1)
    experiment_run = args.experiment_run[0]
    run = args.run[0]
    print('--------------------------')
    print(base_architecture)
    print(pps_per_class)
    print(num_classes)
    print(prototype_shape)
    print(experiment_run)
    print(run)
    print('--------------------------')

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = './saved_models/'+ run + '/' + base_architecture + '_' + experiment_run + '/' 
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settingsMix.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    from settingsMix import train_dir, test_dir, train_push_dir, \
                        train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # all datasets
    # train set
    # Images transformations (preprocess) and data loaders for each part of training and testing
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Pad(50, fill=0, padding_mode="symmetric"),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
            #transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
            transforms.RandomRotation(degrees=(-180, 180)),                     
        ]),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #normalize, #previously images were not normaliced in this step... 
            # I think normalization is necesary, 
            # but it was included in the push funtion (around line 180, part of "push.push_prototypes(")
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    from settingsMix import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settingsMix import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settingsMix import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from settingsMix import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from settingsMix import num_train_epochs, num_warm_epochs, push_start, push_epochs

    # train the model
    log('start training')
    import copy
    prev_accu = 0.0
    prev_push_accu = 0.0
    prev_model_save_path = "./non-exitent_prev_model.pth"
    prev_push_model_save_path = "./non-exitent_prev_push_model.pth"
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        if (prev_accu < accu):
            prev_accu = accu
            delete_previous_models(prev_model_save_path)
            prev_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush_', accu=accu, target_accu=0.0, log=log)
            print(f"saved model at:   {prev_model_save_path}")

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            if (prev_push_accu < accu):
                prev_push_accu = accu       
                delete_previous_models(prev_push_model_save_path)                     
                prev_push_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push_', accu=accu, target_accu=0.0, log=log)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                fc_epochs = 20
                for i in range(fc_epochs):
                    model_was_saved = 0
                    print('epoch: \t{0}'.format(epoch))
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    if (prev_push_accu < accu):
                        prev_push_accu = accu
                        delete_previous_models(prev_push_model_save_path)
                        prev_push_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push_', accu=accu, target_accu=0.0, log=log)
                        model_was_saved = 1
                    #if (i == (fc_epochs - 1)) and (model_was_saved == 0):
                    #     _ = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push_', accu=accu, target_accu=0.0, log=log)
 
    print('--------------------------')
    print(experiment_run)
    print('--------------------------')
    logclose()


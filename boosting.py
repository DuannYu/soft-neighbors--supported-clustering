"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import torch.nn.functional as F

from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.ema import EMA
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.train_utils import boosting_train
from termcolor import colored
from utils.utils import init_experiment

from losses.losses import InstanceLossBoost, ClusterBoostingLoss, ClusterLossBoostV2, SoftInstanceLossBoost, ConfidenceBasedCE



def main():
    device_ids = [0,1,2,3]
    # Retrieve config file
    
    # init experiment
    # Parser
    parser = argparse.ArgumentParser(description='Self-labeling')
    parser.add_argument('--config_env',
                        help='Config file for the environment')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment')
    args = parser.parse_args()
    
    p = create_config(args.config_env, args.config_exp)

    # init experiment
    args.exp_root = './exp_log/'
    args.exp_name = p['train_db_name']
    
    args = init_experiment(args, ['train_vs'])
    
    p['selflabel_checkpoint'] = os.path.join(args.model_dir, 'checkpoint.pth.tar')
    p['selflabel_model'] = os.path.join(args.model_dir, 'model.pth.tar')
    
    
    
    

    
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['scan_model'])
    # print(model)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    strong_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, {'standard': val_transforms, 'augment': strong_transforms},
                                        split='train', to_augmented_dataset=True) 
    train_dataloader = get_train_dataloader(p, train_dataset)
    
    # val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
    if p['train_db_name'] in ['imagenetdogs', 'imagenet10']:
        val_dataset = train_dataset
    else:
        val_dataset = get_val_dataset(p, val_transforms, to_neighbors_dataset = True)
        
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)), 'yellow'))

    # Get criterion
    criterion_ins = SoftInstanceLossBoost()
    criterion_clu = ClusterLossBoostV2(cluster_num=p['num_classes'])
    criterion_clu = ClusterBoostingLoss(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])
    # criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    # Checkpoint
    if os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0

    # EMA
    if p['use_ema']:
        ema = EMA(model, alpha=p['ema_alpha'])
    else:
        ema = None

    # Main loop
    pseudo_labels = -torch.ones(train_dataset.__len__(), dtype=torch.long)

    args.logger.info(colored('Starting main loop', 'blue'))
    
    best_acc = 0.
    for epoch in range(start_epoch, p['epochs']):
        args.logger.info(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Perform self-labeling 
        print('Train ...')
        pseudo_labels = boosting_train(train_dataloader, model, criterion_ins, criterion_clu, pseudo_labels, optimizer, epoch, ema=ema)

        # Update pseudo labels
        
        
        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        predictions = get_predictions(p, val_dataloader, model)
        # import pdb; pdb.set_trace()
        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False) 
        args.logger.info(clustering_stats)
        
        if best_acc < clustering_stats['ACC']:
            print('Best ACC on validation set: %.4f -> %.4f' %(best_acc, clustering_stats['ACC']))
            best_acc = clustering_stats['ACC']
            torch.save(model.module.state_dict(), p['selflabel_model'])
                    
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['selflabel_checkpoint'])
        print('Current Best ACC on validation set is %.4f' % (best_acc))
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions, 
                                # class_names=val_dataset.classes,
                                compute_confusion_matrix=False,
                                confusion_matrix_file=os.path.join(p['selflabel_dir'], 'confusion_matrix.png'))
    args.logger.info(clustering_stats)
    # torch.save(model.module.state_dict(), p['selflabel_model'])


if __name__ == "__main__":
    main()

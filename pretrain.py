import torch
import argparse
from src.models.multiview import load_model, pretrain
from src.datasets.eegdataset import construct_eeg_datasets
from src.datasets.dataset import get_datasets
from torch.optim import AdamW
import os
import wandb
import pickle

def check_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}'):
            i+=1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path

def main(args):
    args.train_mode = 'pretrain'
    # always normalize epochs channelwise within each window
    args.standardize_epochs = 'channelwise'
    
    # load data 
    #pretrain_loader, pretrain_val_loader, _, _, _, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    if not 'sleep' in args.data_path:
        dset = args.data_path.split('/')[-2]
        pretrain_loader, pretrain_val_loader, pretrain_test_loader, (channels, time_length, num_classes) = get_datasets(args.data_path, args.batchsize, pretraining_setup=args.pretraining_setup, combine_all = dset == 'chapman', subsample = False)
    else:
        dset = args.data_path.split('.')[0]
        pretrain_loader, pretrain_val_loader, _, _, _, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
        
    args.orig_channels, args.time_length, args.num_classes = channels, time_length, num_classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = f'pretrained_models/{dset}_{args.model_setup}_{args.pretraining_setup}_{args.loss}'
    print('Saving outputs in', output_path)
    output_path = check_output_path(output_path)

    # initialize wandb
    wandb.init(project = 'MultiView_new', group = f'{dset}_{args.model_setup}_{args.pretraining_setup}', config = args)

    # setup model
    model, loss_fn = load_model(args.model_setup, device, args)

    if args.load_model:
        model.load_state_dict(torch.load(output_path, map_location=device))

    wandb.config.update({'Pretrain samples': len(pretrain_loader.dataset), 'Pretrain validation samples': len(pretrain_val_loader.dataset)})
    
    optimizer = AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

    # pretrain model
    pretrain(model,
            pretrain_loader,
            pretrain_val_loader,
            args.pretrain_epochs,
            optimizer,
            device,
            backup_path=output_path,
            loss_fn = loss_fn,
            log = True)

    model.eval()
    path = f'{output_path}/pretrained_model.pt'
    torch.save(model.state_dict(), path)
    
    # dump arguments to pickle file
    with open(f'{output_path}/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--job_id', type = str, default = '0')
    # whether or not to save finetuned models
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--model_setup', type = str, default = 'MPNN', choices = ['MPNN', 'nonMPNN'])
    parser.add_argument('--pretraining_setup', type = str, default = 'multiview', choices = ['multiview', 'cpc'])
    parser.add_argument('--seed', type = int, default = 42)

    # data arguments
    # path to config files. Remember to change paths in config files. 
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/chapman/') #sleepps18.yml /Users/theb/Desktop/data/HAR/
    parser.add_argument('--finetune_path', type = str, default = 'sleepedf.yml')
    # whether or not to sample balanced during finetuning
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    # number of samples to finetune on. Can be list for multiple runs

    # model arguments
    parser.add_argument('--nlayers', type = int, default = 6)
    # early stopping criterion during finetuning. Can be loss or accuracy (on validation set)
    parser.add_argument('--conv_do', type = float, default = 0.1)
    parser.add_argument('--pool', type = str, default = 'flatten', choices = ['adapt_avg', 'flatten'])
    parser.add_argument('--feat_do', type = float, default = 0.1)
    parser.add_argument('--num_message_passing_rounds', type = int, default = 3)
    parser.add_argument('--hidden_channels', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)
    parser.add_argument('--embedding_dim', type = int, default = 32)


    # eeg arguments
    # subsample number of subjects. If set to False, use all subjects, else set to integer
    parser.add_argument('--sample_pretrain_subjects', type = eval, default = 3)

    # optimizer arguments
    parser.add_argument('--loss', type = str, default = 'time_loss', choices = ['time_loss', 'contrastive', 'COCOA'])
    # whether or not to compute performance on test set during training
    parser.add_argument('--track_test_performance', type = eval, default = True)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--pretrain_epochs', type = int, default = 1)
    parser.add_argument('--batchsize', type = int, default = 128)
    args = parser.parse_args()
    main(args)


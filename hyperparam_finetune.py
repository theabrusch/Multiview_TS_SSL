import torch
import argparse
from src.models.multiview import load_model, finetune, evaluate_classifier
from src.datasets.eegdataset import construct_eeg_datasets
from src.datasets.dataloaders import get_dataloaders_finetuning, get_simulated_data_finetuning
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import wandb
import shutil
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
    args.train_mode = 'finetune'
    # always normalize epochs channelwise within each window
    args.standardize_epochs = 'channelwise'
    
    finetune_loader, finetune_val_loader, test_loader, dset, (channels, time_length, num_classes) = get_dataloaders_finetuning(args, balanced_sampling=args.balanced_sampling, sample_generator=args.sample_generator, seed = args.seed)
    orig_channels = channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if args.load_model:
        pretrained_model_path = f'pretrained_models/{args.pretraining_dset}_{args.model_setup}_{args.pretraining_setup}_{args.loss}'
        output_path = f'finetuned_models/{dset}_{args.model_setup}_{args.pretraining_setup}_{args.loss}'
        group = f'{dset}_{args.model_setup}_{args.pretraining_setup}_{args.loss}' #wandb group
    else:
        output_path = f'finetuned_models/{args.model_setup}_scratch'
        group = f'{dset}_{args.model_setup}_scratch' #wandb group
        # use args from command line
        model_args = args
        model_args.orig_channels = channels
        model_args.time_length = time_length
        model_args.num_classes = num_classes

    output_path = check_output_path(output_path)
    args.output_path = output_path
    print('Saving outputs in', output_path)
    # load model args from pretrained model
    results = {}
    for ft_loader, ft_val_loader in zip(finetune_loader, finetune_val_loader):
        train_samples = len(ft_loader.dataset)
        val_samples = len(ft_val_loader.dataset)
        results[train_samples] = {
            'postfix': [],
            'learning_rate': [],
            'accuracy': [],
        }
        for postfix in args.model_postfix:
            for learning_rate in args.ft_learning_rate:
                group += f'{postfix}_{learning_rate}'
                if args.log:
                    wandb.init(project = 'MultiView_hyperparams', group = group, config = args)
                    wandb.config.update({'Finetune samples': train_samples, 'Finetune validation samples': val_samples, 'Test samples': len(test_loader.dataset)})
                # make sure to save outputs in a new folder
                ft_output_path = output_path + f'/{train_samples}_samples/{postfix}_{learning_rate}'
                os.makedirs(ft_output_path, exist_ok=True)

                # load model with postfix
                pretrained_model_path = f'pretrained_models/{args.pretraining_dset}_{args.model_setup}_{args.pretraining_setup}_{args.loss}{postfix}'

                if args.load_model:
                    model_arg_path = pretrained_model_path + '/args.pkl'
                    with open(model_arg_path, 'rb') as f:
                        model_args = pickle.load(f) 

                # load model
                model = load_model(device, model_args, return_loss=False)
                if args.load_model:
                    pretrained_model_path = pretrained_model_path + '/pretrained_model.pt'
                    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
                
                # update model parameters for finetuning
                model.remove_projector()
                if args.remove_mpnn: # remove the message passing network
                    model.mpnn = False
                    args.optimize_mpnn = False
                # update the classifier to the number of classes in the finetuning dataset
                model.update_classifier(num_classes, orig_channels=orig_channels, pool = args.pool, seed = args.seed)
                # freeze parameters
                model.freeze_parameters(optimize_encoder=args.optimize_encoder, optimize_mpnn=args.optimize_mpnn)

                model.to(device)
                # only optimize parameters that require grad
                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, weight_decay=args.weight_decay)
                
                val_acc = finetune(model,
                                    ft_loader,
                                    ft_val_loader,
                                    args.finetune_epochs,
                                    optimizer,
                                    None,
                                    device,
                                    test_loader = test_loader if args.track_test_performance else None,
                                    early_stopping_criterion=args.early_stopping_criterion,
                                    backup_path=output_path,
                                    log = args.log,
                                    return_score= True,
                )
                # save model
                torch.save(model.state_dict(), f'{ft_output_path}/finetuned_model.pt')
                with open(f'{ft_output_path}/args.pkl', 'wb') as f:
                    pickle.dump(model_args, f)
                results[train_samples]['postfix'].append(postfix)
                results[train_samples]['learning_rate'].append(learning_rate)
                results[train_samples]['accuracy'].append(val_acc)
        best_model = np.argmax(results[train_samples]['accuracy'])
        best_postfix = results[train_samples]['postfix'][best_model]
        best_learning_rate = results[train_samples]['learning_rate'][best_model]
        results[train_samples]['best_postfix'] = best_postfix
        results[train_samples]['best_learning_rate'] = best_learning_rate
        results[train_samples]['best_val_acc'] = results[train_samples]['accuracy'][best_model]
        print(f'Best model for {train_samples} samples: {best_postfix} with learning rate {best_learning_rate}')

        # load model with best postfix
        best_model_path = output_path + f'/{train_samples}_samples/{postfix}_{learning_rate}'
        model_arg_path = best_model_path + '/args.pkl'
        with open(model_arg_path, 'rb') as f:
            model_args = pickle.load(f)
        model = load_model(device, model_args, return_loss=False)
        model.remove_projector()
        if args.remove_mpnn: # remove the message passing network
            model.mpnn = False
            args.optimize_mpnn = False
        # update the classifier to the number of classes in the finetuning dataset
        model.update_classifier(num_classes, orig_channels=orig_channels, pool = args.pool, seed = args.seed)
        model.load_state_dict(torch.load(best_model_path + '/finetuned_model.pt', map_location=device))


        accuracy, prec, rec, f, auc = evaluate_classifier(model, test_loader, device)
        results[train_samples]['test_accuracy'] = accuracy
        results[train_samples]['test_precision'] = prec
        results[train_samples]['test_recall'] = rec
        results[train_samples]['test_f1'] = f
        results[train_samples]['test_auc'] = auc

        if not args.save_model:
            # delete ft_output_path folder to save memory
            shutil.rmtree(output_path)
        # save results file
        res_path = f'outputs/{args.pretraining_dset}_{args.pretraining_setup}_{args.loss}{args.model_postfix}_{dset}_{args.seed}_results.pkl'
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        if args.log:
            wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f, 'Test auc': auc})
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--job_id', type = str, default = '0')
    parser.add_argument('--log', type = eval, default = False) # whether or not to log to wandb
    # whether or not to save finetuned models<<
    parser.add_argument('--save_model', type = eval, default = False)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--optimize_encoder', type = eval, default = False)
    parser.add_argument('--optimize_mpnn', type = eval, default = False)
    parser.add_argument('--pretraining_dset', type = str, default = 'HAR')
    parser.add_argument('--pretraining_setup', type = str, default = 'multiview', choices = ['multiview', 'cpc'])
    parser.add_argument('--model_setup', type = str, default = 'MPNN', choices = ['MPNN', 'nonMPNN', 'average'])
    parser.add_argument('--model_postfix', type = str, nargs = '+', default = [''])

    parser.add_argument('--seed', type = int, default = 42)

    # data arguments
    # path to config files. Remember to change paths in config files. 
    parser.add_argument('--data_path', type = str, default = 'sleepedf_local.yml')
    # whether or not to sample balanced during finetuning
    parser.add_argument('--balanced_sampling', type = eval, default = True)
    # number of samples to finetune on. Can be list for multiple runs
    parser.add_argument('--sample_generator', type = eval, nargs = '+', default = [10, 20])

    # model arguments
    parser.add_argument('--remove_mpnn', type = eval, default = False)
    parser.add_argument('--nlayers', type = int, default = 3)
    # early stopping criterion during finetuning. Can be loss or accuracy (on validation set)
    parser.add_argument('--early_stopping_criterion', type = str, default = None, choices = [None, 'loss', 'acc'])
    parser.add_argument('--pool', type = str, default = 'adapt_avg', choices = ['adapt_avg', 'flatten'])
    parser.add_argument('--conv_do', type = float, default = 0.1)
    parser.add_argument('--feat_do', type = float, default = 0.1)
    parser.add_argument('--num_message_passing_rounds', type = int, default = 3)
    parser.add_argument('--hidden_channels', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)
    parser.add_argument('--embedding_dim', type = int, default = 32)


    # eeg arguments
    # subsample number of subjects. If set to False, use all subjects, else set to integer
    parser.add_argument('--sample_finetune_train_subjects', type = eval, default = 2)
    parser.add_argument('--sample_finetune_val_subjects', type = eval, default = 2)
    parser.add_argument('--sample_test_subjects', type = eval, default = 2)

    # optimizer arguments
    parser.add_argument('--loss', type = str, default = 'contrastive', choices = ['time_loss', 'contrastive', 'COCOA'])
    # whether or not to compute performance on test set during training
    parser.add_argument('--track_test_performance', type = eval, default = False)
    parser.add_argument('--ft_learning_rate', type = float, nargs = '+', default = [1e-3, 5e-4])
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--batchsize', type = int, default = 128)
    args = parser.parse_args()
    main(args)


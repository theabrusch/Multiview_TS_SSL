import torch
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from src.models import ContrastiveLoss2
from copy import deepcopy
import wandb

def TFC_trainer(model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                epochs, 
                val_loader, 
                device, 
                contrastive_encoding = 'all',
                backup_path = None,
                log = True,
                classifier = None,
                class_optimizer = None, 
                delta_ = 1, 
                lambda_ = 0.5, 
                eta_ = 0.5):
    """Function for training the time frequency contrastive model for time series. 

    Args:
        model (torch.Module): The model to train
        train_loader (torch.utils.data.DataLoader): Dataloader containing the train data on which to train the model
        optimizer (torch.optim.Optimizer): Optimizer with which to optimize the model. 
        loss_fn (torch.Module): Function implementing the contrastive loss function to use for 
                                optimizing the self-supervised part of the model.
        epochs (int): Number of epochs to train the model for. 
        val_loader (torch.utils.data.DataLoader): Dataloader containing the validation data on which to validate the model
        device (torch.device): CPU or GPU depending on availability
        train_classifier (bool): Whether to train the classifier along with the contrastive loss. 
        delta_ (int, optional): Parameter to add in the time frequency consistency loss. Defaults to 1.
        lambda_ (float, optional): Parameter weighing the time and frequency loss vs the time-frequency consistency loss. Defaults to 0.5.
        eta_ (float, optional): Parameter weighing the contrastive loss vs the classifier. Defaults to 0.5.

    Returns:
    torch.Module: Final model after training
    dict: Dictionary containing all of the losses
    """    
    time_loss_total = []
    freq_loss_total = []
    time_freq_loss_total = []
    loss_total = []
    val_time_loss_total = []
    val_freq_loss_total = []
    val_time_freq_loss_total = []
    val_loss_total = []

    if classifier is not None:
        class_loss_fn = torch.nn.CrossEntropyLoss()
        class_loss_total = []
        val_class_loss_total = []

    for epoch in range(epochs):
        print('\n', epoch + 1 , 'of', epochs)
        epoch_time, epoch_freq, epoch_time_freq, epoch_class, epoch_loss = [0, 0, 0, 0, 0]
        val_epoch_time, val_epoch_freq, val_epoch_time_freq, val_epoch_class, val_epoch_loss, val_epoch_acc = [0, 0, 0, 0, 0, 0]
        model.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)

            h_t, z_t, h_f, z_f = model(x_t, x_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)            
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + delta_) + (time_freq_pos - time_freq_neg[1] + delta_) + (time_freq_pos - time_freq_neg[2] + delta_)

            if classifier is not None:
                classifier.train()
                y_out = classifier(torch.cat([z_t, z_f], dim = -1))
                class_loss = class_loss_fn(y_out, y)
                loss = eta_*class_loss + (1-eta_)*(lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC)
                epoch_class += class_loss.detach().cpu()
            else:
                if contrastive_encoding == 'time':
                    loss = time_loss
                elif contrastive_encoding == 'freq':
                    loss = freq_loss
                elif contrastive_encoding == 'time_freq':
                    loss = time_loss + freq_loss
                elif contrastive_encoding == 'TFC':
                    loss = loss_TFC
                elif contrastive_encoding == 'all':
                    loss = lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC

            epoch_time += time_loss.detach().cpu()
            epoch_freq += freq_loss.detach().cpu()
            epoch_time_freq += loss_TFC.detach().cpu()
            epoch_loss += loss.detach().cpu()

            loss.backward()
            optimizer.step()
            if classifier is not None:
                class_optimizer.step()
        
        print('\nTraining losses:')
        print('Time consistency loss:', epoch_time/(i+1))
        print('Frequency consistency loss:', epoch_freq/(i+1))
        print('Time-freq consistency loss:', epoch_time_freq/(i+1))
        print('Total loss:', epoch_loss/(i+1))

        if log:
            wandb.log({'pretrain time loss': epoch_time/(i+1), 'pretrain freq loss': epoch_freq/(i+1), 'pretrain TFC': epoch_time_freq/(i+1), 'pretrain total loss': epoch_loss/(i+1)})

        if not backup_path is None:
            path = backup_path
            torch.save(model.state_dict(), path)
            
        time_loss_total.append(epoch_time/(i+1))
        freq_loss_total.append(epoch_freq/(i+1))
        time_freq_loss_total.append(epoch_time_freq/(i+1))
        loss_total.append(epoch_loss/(i+1))
        if classifier is not None:
            class_loss_total.append(epoch_class/(i+1))
            print('Classification loss:', epoch_class/(i+1))

        # evaluate on validation set
        model.eval()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)
            h_t, z_t, h_f, z_f = model(x_t, x_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + 1) + (time_freq_pos - time_freq_neg[1] + 1) + (time_freq_pos - time_freq_neg[2] + 1)

            loss = lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC

            if classifier is not None:
                classifier.eval()
                y_out = classifier(torch.cat([z_t, z_f], dim = -1))
                class_loss = class_loss_fn(y_out, y)
                loss += class_loss
                val_epoch_class += class_loss.detach().cpu()

                if i == 0:
                    y_pred = torch.argmax(y_out.detach().cpu(), dim = 1)
                    y_true = y.detach().cpu()
                else:
                    y_pred = torch.cat((y_pred, torch.argmax(y_out.detach().cpu(), dim = 1)), dim = 0)
                    y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)
                

            val_epoch_time += time_loss.detach().cpu()
            val_epoch_freq += freq_loss.detach().cpu()
            val_epoch_time_freq += loss_TFC.detach().cpu()
            val_epoch_loss += loss.detach().cpu()
        
        print('\nValidation losses')
        print('Time consistency loss:', val_epoch_time/(i+1))
        print('Frequency consistency loss:', val_epoch_freq/(i+1))
        print('Time-freq consistency loss:', val_epoch_time_freq/(i+1))
        print('Total loss:', val_epoch_loss/(i+1))

        if log:
            wandb.log({'pretrain val time': val_epoch_time/(i+1), 'pretrain val freq': val_epoch_freq/(i+1), 'pretrain val TFC': val_epoch_time_freq/(i+1), 'pretrain val total': val_epoch_loss/(i+1)})
        
        val_time_loss_total.append(val_epoch_time/(i+1))
        val_freq_loss_total.append(val_epoch_freq/(i+1))
        val_time_freq_loss_total.append(val_epoch_time_freq/(i+1))
        val_loss_total.append(val_epoch_loss/(i+1))
        if classifier is not None:
            val_class_loss_total.append(val_epoch_class/(i+1))
            acc = balanced_accuracy_score(y_true, y_pred)
            print('Accuracy:', acc)
            print('Classification loss:', val_epoch_class/(i+1))



    losses = {
        'train': {
            'time_loss': time_loss_total,
            'freq_loss': freq_loss_total,
            'time_freq_loss': time_freq_loss_total,
            'loss': loss_total},
        'val': {
            'time_loss': val_time_loss_total,
            'freq_loss': val_freq_loss_total,
            'time_freq_loss': val_time_freq_loss_total,
            'loss': val_loss_total}
        }
    if classifier is not None:
        losses['train']['class_loss'] = class_loss_total
        losses['val']['class_loss'] = val_class_loss_total

    return model, losses

def train_classifier(model, 
                    train_loader, 
                    optimizer, 
                    epochs, 
                    val_loader, 
                    device):
    """Function for training only the classifier part of the TFC model. 

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        optimizer (_type_): _description_
        epochs (_type_): _description_
        val_loader (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    
    loss_total = []
    val_loss_total = []

    class_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('\n', epoch + 1 , 'of', epochs)
        epoch_loss = 0
        epoch_acc = 0 
        val_epoch_loss = 0
        val_epoch_acc = 0 
        y_pred = []
        y_true = []
        model.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)

            _, _, _, _, out = model(x_t, x_f)

            class_loss = class_loss_fn(out, y)
                
            epoch_loss += class_loss.detach().cpu()

            if i == 0:
                y_pred = torch.argmax(out.detach().cpu(), dim = 1)
                y_true = y.detach().cpu()
            else:
                y_pred = torch.cat((y_pred, torch.argmax(out.detach().cpu(), dim = 1)), dim = 0)
                y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)

            class_loss.backward()
            optimizer.step()
        
        epoch_acc += balanced_accuracy_score(y_true, y_pred)
        print('\nTraining losses:')
        print('Accuracy', epoch_acc)
        print('Total loss:', epoch_loss/(i+1))

        loss_total.append(epoch_loss/(i+1))

        # evaluate on validation set
        model.eval()
        y_pred = []
        y_true = []
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long()
            _, _, _, _, out = model(x_t, x_f)

            class_loss = class_loss_fn(out.detach().cpu(), y)

            if i == 0:
                y_pred = torch.argmax(out.detach().cpu(), dim = 1)
                y_true = y.detach().cpu()
            else:
                y_pred = torch.cat((y_pred, torch.argmax(out.detach().cpu(), dim = 1)), dim = 0)
                y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)
            val_epoch_loss += class_loss.detach().cpu()
        
        val_epoch_acc += balanced_accuracy_score(y_true, y_pred)
        
        print('\nValidation losses')
        print('Accuracy:', val_epoch_acc)
        print('Total loss:', val_epoch_loss/(i+1))
    
        val_loss_total.append(val_epoch_loss/(i+1))



    losses = {
        'train': {
            'loss': loss_total},
        'val': {
            'loss': val_loss_total}
        }
    return model, losses

def finetune_model(model, 
                  classifier, 
                  data_loader, 
                  val_loader,
                  loss_fn, 
                  optimizer, 
                  class_optimizer, 
                  epochs, 
                  device,
                  return_best = False,
                  writer = None, 
                  delta = 0.5, 
                  lambda_ = 0.5):

    model.train()
    classifier.train()
    class_loss_fn = torch.nn.CrossEntropyLoss()

    collect_class_loss = torch.zeros(epochs)
    collect_loss = torch.zeros(epochs)
    collect_time_loss = torch.zeros(epochs)
    collect_freq_loss = torch.zeros(epochs)
    collect_time_freq_loss = torch.zeros(epochs)

    collect_val_class_loss = torch.zeros(epochs)
    collect_val_loss = torch.zeros(epochs)
    collect_val_time_loss = torch.zeros(epochs)
    collect_val_freq_loss = torch.zeros(epochs)
    collect_val_time_freq_loss = torch.zeros(epochs)
    accuracy = 0
    best_state_dict = deepcopy(model.state_dict())
    best_class_state_dict = deepcopy(classifier.state_dict())

    for epoch in range(epochs):
        print('\n', epoch + 1 , 'of', epochs)
        epoch_loss = 0
        epoch_class_loss = 0
        epoch_time_loss = 0
        epoch_freq_loss = 0
        epoch_time_freq_loss = 0
        model.train()
        classifier.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(data_loader):
            if optimizer is not None:
                optimizer.zero_grad()
            class_optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)
            h_t, z_t, h_f, z_f = model(x_t, x_f, finetune = True)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug, finetune = True)

            time_loss = loss_fn(h_t, h_t_aug)
            #time_loss = 0
            #freq_loss = 0
            #loss_TFC = 0
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + 1) + (time_freq_pos - time_freq_neg[1] + 1) + (time_freq_pos - time_freq_neg[2] + 1)
            y_out = classifier(torch.cat([z_t, z_f], dim = -1))
            
            class_loss = class_loss_fn(y_out, y)
            loss = delta*class_loss + (1-delta)*(lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            class_optimizer.step()
            epoch_loss += loss.detach().cpu()
            epoch_class_loss += class_loss.detach().cpu()
            epoch_time_loss += time_loss.detach().cpu()
            epoch_freq_loss += freq_loss.detach().cpu()
            epoch_time_freq_loss += loss_TFC.detach().cpu()
        
        collect_class_loss[epoch] = epoch_class_loss / (i+1)
        collect_loss[epoch] = epoch_loss / (i+1)
        collect_time_loss[epoch] = epoch_time_loss / (i+1)
        collect_freq_loss[epoch] = epoch_freq_loss / (i+1)
        collect_time_freq_loss[epoch] = epoch_time_freq_loss / (i+1)
        print('\nTraining loss')
        print('Epoch loss:', epoch_loss/(i+1))
        print('Class. loss:', epoch_class_loss/(i+1))
        if not writer is None:
            writer.add_scalar('train_finetune/class_loss', epoch_class_loss / (i+1), epoch)
            writer.add_scalar('train_finetune/total_loss', epoch_loss / (i+1), epoch)
            writer.add_scalar('train_finetune/time_loss', epoch_time_loss/(i+1), epoch)
            writer.add_scalar('train_finetune/time_freq_loss', epoch_time_freq_loss/(i+1), epoch)
            writer.add_scalar('train_finetune/freq_loss', epoch_freq_loss/(i+1), epoch)

        epoch_loss = 0
        epoch_class_loss = 0
        epoch_time_loss = 0
        epoch_freq_loss = 0
        epoch_time_freq_loss = 0

        model.eval()
        classifier.eval()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)
            h_t, z_t, h_f, z_f = model(x_t, x_f, finetune = True)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug, finetune = True)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + 1) + (time_freq_pos - time_freq_neg[1] + 1) + (time_freq_pos - time_freq_neg[2] + 1)

            y_out = classifier(torch.cat([z_t, z_f], dim = -1))
            class_loss = class_loss_fn(y_out, y)
            loss = delta*class_loss + (1-delta)*(loss_TFC)

            epoch_loss += loss.detach().cpu()
            epoch_class_loss += class_loss.detach().cpu()
            epoch_time_loss += time_loss.detach().cpu()
            epoch_freq_loss += freq_loss.detach().cpu()
            epoch_time_freq_loss += loss_TFC.detach().cpu()
        
        print('\nValidation loss')
        print('Epoch loss:', epoch_loss/(i+1))
        print('Class. loss:', epoch_class_loss/(i+1))

        collect_val_class_loss[epoch] = epoch_class_loss / (i+1)
        collect_val_loss[epoch] = epoch_loss / (i+1)
        collect_val_time_loss[epoch] = epoch_time_loss / (i+1)
        collect_val_freq_loss[epoch] = epoch_freq_loss / (i+1)
        collect_val_time_freq_loss[epoch] = epoch_time_freq_loss / (i+1)

        if not writer is None:
            writer.add_scalar('val_finetune/class_loss', epoch_class_loss / (i+1), epoch)
            writer.add_scalar('val_finetune/total_loss', epoch_loss / (i+1), epoch)
            writer.add_scalar('val_finetune/time_loss', epoch_time_loss/(i+1), epoch)
            writer.add_scalar('val_finetune/time_freq_loss', epoch_time_freq_loss/(i+1), epoch)
            writer.add_scalar('val_finetune/freq_loss', epoch_freq_loss/(i+1), epoch)

        results = evaluate_model(model, classifier, val_loader, device)
        print('Validation accuracy:', results['Accuracy'])
        print('Validation precision:', np.mean(results['Precision']))
        print('Validation recall:', np.mean(results['Recall']))
        print('Validation F1:', np.mean(results['F1 score']))

        if not writer is None:
            writer.add_scalar('val_finetune/accuracy', results['Accuracy'], epoch)
            writer.add_scalar('val_finetune/precision', np.mean(results['Precision']), epoch)
            writer.add_scalar('val_finetune/recall', np.mean(results['Recall']), epoch)
            writer.add_scalar('val_finetune/f1', np.mean(results['F1 score']), epoch)

        if return_best:
            if results['Accuracy'] > accuracy:
                best_state_dict = deepcopy(model.state_dict())
                best_class_state_dict = deepcopy(classifier.state_dict())
                accuracy = results['Accuracy']
    
    losses = {'train': {
        'Loss': collect_loss,
        'Class loss': collect_class_loss,
        },
        'val': {
        'Loss': collect_val_loss,
        'Class loss': collect_val_class_loss,
        }
    }

    if return_best:
        model.load_state_dict(best_state_dict)
        classifier.load_state_dict(best_class_state_dict)

    return model, classifier, losses

    
def evaluate_model(model,
                   classifier,
                   test_loader,
                   device):
    
    model.eval()
    classifier.eval()
    for i, variables in enumerate(test_loader):
        x_t, x_f, y = variables[0], variables[1], variables[-1]
        x_t, x_f, y = x_t.float().to(device), x_f.float().to(device), y.long()
        _, z_t, _, z_f = model(x_t, x_f, finetune = True)
        y_out = classifier(torch.cat([z_t, z_f], dim = -1))

        if i == 0:
            y_pred = torch.argmax(y_out, dim = -1).detach().cpu()
            y_true = y
        else:
            y_pred = torch.cat([y_pred, torch.argmax(y_out, dim = -1).detach().cpu()], dim = 0)
            y_true = torch.cat([y_true, y], dim = 0)
    
    acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f, _ = precision_recall_fscore_support(y_true, y_pred)
    #auroc = roc_auc_score(y_true, y_pred)
    #auprc = average_precision_score(y_true, y_pred)

    results = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 score': f
        #'AUROC': auroc, 
        #'AUPRC': auprc
    }

    for res in results.keys():
        print(res, ':', results[res])
    
    return results






def evaluate_latent_space(model, data_loader, device, classifier = False, save_h = True):
    model.eval()
    loss_fn = ContrastiveLoss2(tau = 0.2, device = device, reduce = False)
    collect_z_latent_space = []
    collect_y = []
    collect_h_losses = []
    collect_z_losses = []
    for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(data_loader):
        x_t, x_f, x_t_aug, x_f_aug = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device)
        normal_outputs = model(x_t, x_f)
        augmented_outputs = model(x_t_aug, x_f_aug)
        time_loss = loss_fn(normal_outputs[0], augmented_outputs[0]).unsqueeze(0)
        freq_loss = loss_fn(normal_outputs[2], augmented_outputs[2]).unsqueeze(0)

        time_freq_pos = loss_fn(normal_outputs[1], normal_outputs[3]).unsqueeze(0)
        time_freq_neg = []
        for pair in [[normal_outputs[1], augmented_outputs[3]], [augmented_outputs[1], normal_outputs[3]], [augmented_outputs[1], augmented_outputs[3]]]:
            time_freq_neg.append(loss_fn(*pair).unsqueeze(0))

        normal_outputs = [out.detach().cpu().numpy() for out in normal_outputs]
        augmented_outputs = [out.detach().cpu().numpy() for out in augmented_outputs]
        if save_h:
            h_latent_space = np.concatenate((normal_outputs[0][np.newaxis,:, :], normal_outputs[2][np.newaxis, :, :], augmented_outputs[0][np.newaxis, :, :], augmented_outputs[2][np.newaxis, :, :]), axis = 0)
        z_latent_space = np.concatenate((normal_outputs[1][np.newaxis, :, :], normal_outputs[3][np.newaxis, :, :], augmented_outputs[1][np.newaxis, :, :], augmented_outputs[3][np.newaxis, :, :]), axis = 0)

        if save_h:
            collect_h_latent_space = np.concatenate((collect_h_latent_space, h_latent_space), axis = 1)
        collect_z_latent_space.append(z_latent_space)
        collect_y.append(y.numpy())
        #collect_x_t = np.concatenate((collect_x_t, x_t.detach().cpu().numpy()), axis = 0)
        collect_h_losses.append(torch.cat((time_loss, freq_loss), dim = 0).detach().cpu().numpy())
        collect_z_losses.append(torch.cat((time_freq_pos,*time_freq_neg), dim = 0).detach().cpu().numpy())
        if classifier:
            collect_y_out = np.concatenate((collect_y_out, normal_outputs[-1]), axis = 0)
    
    columns_h = ['h_t', 'h_f', 'h_t_aug', 'h_f_aug'] 
    columns_z = ['z_t', 'z_f', 'z_t_aug', 'z_f_aug'] 
    outputs = dict()

    if save_h:
        output_columns = [zip(columns_h, collect_h_latent_space), zip(columns_z, collect_z_latent_space)]
    else:
        output_columns = [zip(columns_z, np.hstack(collect_z_latent_space))]

    for latent in output_columns:
        for i, (name, var) in enumerate(latent):
            outputs[name] = var
            
    outputs['y'] = np.hstack(collect_y)
    #outputs['x'] = collect_x_t
    outputs['z_losses'] = np.hstack(collect_z_losses)
    outputs['h_losses'] = np.hstack(collect_h_losses)
    
    if classifier:
        outputs['y_pred'] = collect_y_out

    return outputs


    
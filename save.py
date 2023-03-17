import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if (accu > target_accu):
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
    save_path_and_model_name = os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu))
    torch.save(obj=model, f=save_path_and_model_name)
    #print(f"saved model at:   {save_path_and_model_name}")
    return save_path_and_model_name

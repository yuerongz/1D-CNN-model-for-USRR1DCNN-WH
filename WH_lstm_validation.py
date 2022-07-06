from WH_lstm_model import LSTMModel
from WH_1dcnn_training import DataBatcher
from base_func import *
import numpy as np
import torch
import torch.optim as optim
import pandas as pd


def load_saved_lstm_model(model_file, epoch_needed=False):
    device = pytorch_cuda_check()
    checkpoint = torch.load(model_file)
    model = LSTMModel(checkpoint['model_structure']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['loss_his']
    val_losses = checkpoint['val_loss_his']
    eval_losses = checkpoint['eval_losses']
    eval_val_losses = checkpoint['eval_val_losses']
    model.eval()
    if epoch_needed:
        return model, losses, val_losses, eval_losses, eval_val_losses, epoch, checkpoint['seq_h']
    else:
        return model, losses, val_losses, eval_losses, eval_val_losses


def speed_test(work_dir):
    import glob
    import time
    file_ls = glob.glob(f"{work_dir[:-1]}/model_*.pt")
    device = pytorch_cuda_check()
    validation_dataset = DataBatcher(None, input_time_len_h=96, rl_group=None, validation=True)
    validation_inputs = validation_dataset.validation_inputs
    val_in = validation_inputs.to(device).float()# .reshape(4, -1, *validation_inputs.shape[-2:])[0]
    pred_time = []
    sttm0 = time.time()
    for file_name in file_ls:
        model_specs = load_saved_lstm_model(file_name, True)
        model = model_specs[0]
        model.eval()
        with torch.no_grad():
            sttm = time.time()
            _ = model(val_in, None, None, False)
            pred_time.append(time.time() - sttm)
    return print(np.sum(np.array(pred_time)), time.time() - sttm0)


def validation_preds(work_dir):
    import glob
    from gdal_func import gdal_writetiff
    ## save raw prediction of models for validation dataset
    file_ls = glob.glob(f"{work_dir[:-1]}/model_*.pt")
    device = pytorch_cuda_check()
    for file_name in file_ls:
        group_i = int(file_name.split('_')[4])
        model_specs = load_saved_lstm_model(file_name, True)
        validation_dataset = DataBatcher(None, input_time_len_h=model_specs[-1], rl_group=group_i, validation=True)
        validation_inputs = validation_dataset.validation_inputs
        val_in = validation_inputs.to(device).float()#.reshape(4, -1, *validation_inputs.shape[-2:])
        model = model_specs[0]
        model.eval()
        with torch.no_grad():
            pred = model(val_in, None, None, False)
            gdal_writetiff(pred.detach().cpu().numpy(),
                           f"{work_dir}unfilled_pred_depths_{group_i}.tif", target_transform=(0, 1, 0, 0, 0, -1))
    # ## save depths (re-ordered in RL geo-order)
    device = pytorch_cuda_check()
    group_arr = pd.read_csv('clustering/v1.csv').cluster_i.to_numpy()
    pred_depths1 = torch.zeros((2400, group_arr.shape[0]), requires_grad=False).to(device).float()
    for group_i in range(423):
        curr_gp = torch.from_numpy(gdal_asarray(f"{work_dir}unfilled_pred_depths_{group_i}.tif")).to(device).float()
        pred_depths1[:, group_arr == group_i] = curr_gp
    pred_depths1 = pred_depths1.detach().cpu().numpy()
    gdal_writetiff(pred_depths1, f"{work_dir}final_depth_predictions_unfilled.tif",
                   target_transform=(0, 1, 0, 0, 0, -1))
    validation_dataset = DataBatcher(None, input_time_len_h=96, validation=True)
    validation_refs = validation_dataset.rl_dataset_prep()
    gdal_writetiff(validation_refs.detach().cpu().numpy(), f"{work_dir}final_depth_predictions_ref.tif",
                   target_transform=(0, 1, 0, 0, 0, -1))
    ## example of use:
    # ref = gdal_asarray(f"{work_dir}final_depth_predictions_ref.tif")
    # pred_depths1 = gdal_asarray(f"{work_dir}final_depth_predictions_unfilled.tif")
    # pred_depths1[pred_depths1 < 0] = 0
    return 0


if __name__ == '__main__':
    work_dir = 'lstm_results/group_test6/'
    # speed_test(work_dir)
    validation_preds(work_dir)

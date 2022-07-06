from base_func import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from WH_1dcnn_model import ConvoModel2RLs


class DataBatcher:
    def __init__(self, batch_size, validation=False, input_time_len_h=96, time_lag_h=0,
                 rl_group=None, timestep=3, development_mode=False):
        self.platform_defaults()

        self.validation_mode = validation
        self.input_seq_len = int(input_time_len_h * 12 / timestep)
        self.input_seq_start = int(time_lag_h * 12 / timestep) + self.input_seq_len
        event_log, val_event_log = get_tf_event_log()
        if validation:
            event_log = val_event_log
        self.event_log = event_log
        self.timestep = timestep
        self.time_slicing = lambda x: x[self.timestep//2::self.timestep]
        self.file_naming = lambda evt_info: f'{self.data_dir}ts_depth_{evt_info.eventname}.tif'
        self.rl_filter_prep(rl_group)
        if (not validation) & (not development_mode):
            self.train_loader, self.test_loader = self.get_RL_batch_loaders(batch_size)
        elif not development_mode:
            self.validation_inputs = self.input_tensor_prep()

    def platform_defaults(self):
        if platform == 'linux':  # on Spartan
            prj_dir = '/home/yuerongz/punim0728/WHProj/'
            self.nc_dir = '/home/yuerongz/scratch_prj0728/TUFLOW_WH_yz/results/'
            self.data_dir = f'{prj_dir}preprocess_data/'
            self.pts_file = f'{prj_dir}srr_results/ss_100.shp'
            self.dem_file = f'{prj_dir}gis/dem_croped.tif'
            self.input_dir = '/home/yuerongz/punim0728/TUFLOW_WH_yz/bc_dbase/'

        elif platform == 'win32':
            self.nc_dir = '../TUFLOW_WH_yz/results/'
            self.data_dir = 'preprocess_data/'
            self.pts_file = 'srr_results/ss_100.shp'
            self.dem_file = 'gis/dem_croped.tif'
            self.rl_grouping_file = 'clustering/v1.csv'
            self.input_dir = '../TUFLOW_WH_yz/bc_dbase/'
        else:
            raise SystemError('Please set your own system defaults in DataBatcher.platform_defaults().')
        return 0

    def rl_filter_prep(self, rl_group):
        if rl_group is None:
            self.rl_filter = lambda x: x
        else:
            rl_group_arr = pd.read_csv(self.rl_grouping_file).cluster_i.to_numpy()
            self.rl_num = np.sum(rl_group_arr == rl_group)
            self.rl_filter = lambda x: x[:, rl_group_arr == rl_group]
        return 0

    def rl_dataset_prep(self):
        """ Read in all 50 events data."""
        rl_curr = self.rl_filter(self.time_slicing(gdal_asarray(self.file_naming(self.event_log.iloc[0]))))
        dataset_arr = rl_curr   #(1801, 21133)
        for evt_id in range(1, len(self.event_log)):
            rl_curr = self.rl_filter(self.time_slicing(gdal_asarray(self.file_naming(self.event_log.iloc[evt_id]))))
            dataset_arr = np.append(dataset_arr, rl_curr, axis=0)
        return torch.from_numpy(dataset_arr)

    def data_splitting(self, tensor_ls, batch_size, split_ratio, shuffle, random_seed):
        idx_expander = lambda x, mul_i: np.repeat(x, mul_i) * mul_i + np.array(list(range(mul_i)) * len(x))
        test_evt_idxs = np.array([2, 8, 16, 23, 25, 34, 39, 43, 46, 48])
        idxs_per_evt = np.ceil((1801 - self.timestep//2) / self.timestep).astype(int)
        test_map_idxs = idx_expander(test_evt_idxs, idxs_per_evt)
        train_evt_idxs = np.delete(np.arange(50), test_evt_idxs)
        train_map_idxs = idx_expander(train_evt_idxs, idxs_per_evt)
        rng = np.random.default_rng(random_seed)
        if shuffle:
            rng.shuffle(train_map_idxs)
            rng.shuffle(test_map_idxs)
        train_data_ls = [torch.split(tensor_i[train_map_idxs.astype(int)], batch_size) for tensor_i in tensor_ls]
        test_data_ls = [torch.split(tensor_i[test_map_idxs.astype(int)], batch_size) for tensor_i in tensor_ls]
        if len(tensor_ls) > 1:
            train_loader = list(zip(*train_data_ls))
            test_loader = list(zip(*test_data_ls))
        else:
            train_loader = train_data_ls[0]
            test_loader = test_data_ls[0]
        return train_loader, test_loader

    def get_RL_batch_loaders(self, batch_size, split_ratio=0.8, shuffle=True, random_seed=321):
        input_tensor = self.input_tensor_prep()
        rls_ts =  self.rl_dataset_prep()
        train_loader, test_loader = self.data_splitting([input_tensor, rls_ts], batch_size, split_ratio, shuffle, random_seed)
        return train_loader, test_loader

    def input_tensor_prep(self):
        seq_len = self.input_seq_len
        # read from TUFLOW input csv    # loop csv files, save all data
        # segmentise data (48-70 hours step), shape=(90050, time_len*12, num_of_inflows[33])
        event_input = []
        for evt_id in range(len(self.event_log)):
            event_name = self.event_log.iloc[evt_id].eventname
            raw_input = get_tf_event_input(self.input_dir, event_name)
            kinginflow_idx = raw_input.columns.to_list().index('KingInflow')# 20
            ovensinflow_idx = raw_input.columns.to_list().index('S117J1_S117J2')# 50
            input_arr = raw_input.to_numpy()
            input_arr = self.time_slicing(input_arr[:, 1:])
            input_arr = np.r_['0,2', np.zeros((self.input_seq_start-1, input_arr.shape[1])), input_arr]
            input_arr[:self.input_seq_start-1, kinginflow_idx] = 20.
            input_arr[:self.input_seq_start-1, ovensinflow_idx] = 50.
            event_input.append([input_arr[i:i+seq_len, :] for i in range(len(self.time_slicing(raw_input)))])
        event_input = np.array(event_input) # shape = (event_num, evt_data_points, seq_len, var_num)
        event_input = event_input.reshape((-1, *event_input.shape[-2:]))
        event_input = event_input / 2519.438034447 # event_input.max()   # rescale to [0, 1]
        #                       (x, 576, 33)
        return torch.from_numpy(event_input)


def training(work_dir, saving_tag, batch_sz, structure_szs, seq_h, rl_group, epoch_len=50, learning_rt=0.001):
    data_loaders = DataBatcher(batch_sz, input_time_len_h=seq_h, rl_group=rl_group, validation=False,
                               development_mode=True)
    rl_size = data_loaders.rl_num
    train_loader, test_loader = data_loaders.data_splitting([data_loaders.input_tensor_prep(),
                                                             data_loaders.rl_dataset_prep()
                                                             ], batch_sz, 0.8, True, 321)
    validation_dataset = DataBatcher(None, input_time_len_h=seq_h, rl_group=rl_group, validation=True)
    validation_inputs = validation_dataset.validation_inputs
    validation_refs = validation_dataset.rl_dataset_prep()
    structure_szs[-1] = rl_size

    device = pytorch_cuda_check()
    losses = []
    eval_losses = []
    val_losses = []
    eval_val_losses = []
    model = ConvoModel2RLs(structure_szs, seq_h).to(device)
    model.float()
    loss_function = nn.MSELoss()
    eval_loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rt)  # weight_decay=0.00001 tested. Not desired.

    sttm = time.time()
    # train the model
    for epoch in range(epoch_len):
        for input_in_batch, output_in_batch in train_loader:
            input_in_batch = input_in_batch.to(device)
            output_in_batch = output_in_batch.to(device)
            model.train()  # set model to TRAIN mode
            optimizer.zero_grad()
            pred = model(input_in_batch.float(), None, output_in_batch.float(), False)
            out_ref = output_in_batch.float()
            loss = loss_function(pred, out_ref)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            eval_losses.append(eval_loss_function(pred, out_ref).detach().cpu().numpy())

        with torch.no_grad():
            for input_in_batch, output_in_batch in test_loader:
                input_in_batch = input_in_batch.to(device)
                output_in_batch = output_in_batch.to(device)
                model.eval()  # set model to TEST mode
                pred = model(input_in_batch.float(), None, output_in_batch.float(), False)
                out_ref = output_in_batch.float()
                val_loss = loss_function(pred, out_ref)
                val_losses.append(val_loss.item())
                eval_val_losses.append(eval_loss_function(pred, out_ref).detach().cpu().numpy())
            y_hat = model(validation_inputs.to(device).float(), None, None, False)
            final_loss = loss_function(y_hat, validation_refs.to(device).float()).item()
        # print(f"Epoch {epoch}: train loss={np.mean(losses[-len(train_loader):])}, "
        #       f"test loss={np.mean(val_losses[-len(test_loader):])}; "
        #       f"validation loss={final_loss}; "
        #       f"eval_mtx={np.mean(eval_losses[-len(train_loader):])}, "
        #       f"eval_test={np.mean(eval_val_losses[-len(test_loader):])}."
        #       f"time_since_start={time.time()-sttm}s.")
        if (np.mean(val_losses[-len(test_loader):]) <= 0.0015) \
                | ((np.mean(losses[-len(train_loader):]) < 0.004) &
                   (np.mean(val_losses[-len(test_loader):]) > 0.008)):   # RMSE = 3.873cm; early stopping
            break
    print(f"group={rl_group} "
          f"{np.mean(losses[-len(train_loader):])} {np.mean(val_losses[-len(test_loader):])} {final_loss} "
          f"{np.mean(eval_losses[-len(train_loader):])} {np.mean(eval_val_losses[-len(test_loader):])} "
          f"time_since_start={(time.time()-sttm)/60}min.")
    nn_model_training_summary(model, losses, val_losses, eval_losses, eval_val_losses, epoch,
                              fig_title=f'MSE: group = {rl_group}', fig_file=f'{work_dir}{rl_group}.png')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_his': losses,
        'val_loss_his': val_losses,
        'eval_losses': eval_losses,
        'eval_val_losses': eval_val_losses,
        'learning_rate': learning_rt,
        'batch_size': batch_sz,
        'model_structure': structure_szs,
        'seq_h': seq_h
    }, f"{work_dir}model_{saving_tag}_ep{epoch}.pt")

    return model


if __name__ == '__main__':
    work_dir = 'convo_results/group_test3/'
    directory_checking(work_dir)
    batch_sz = 50
    ep = 40
    seq_h = 96
    rl_size = 50    # default; estimation only; placeholder
    model_structure = [33, 32, 64, None, rl_size]   # input_dim, conv2_channel_out_dim, hidden 1-2, rv_no
    for group_i in range(423):
        saving_tag = f'convo_{group_i}'
        model = training(work_dir, saving_tag, batch_sz, model_structure, seq_h, group_i,
                         epoch_len=ep, learning_rt=0.001)


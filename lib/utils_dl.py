import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# set seeds for reproducibility
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


# calculate the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation(str_act, param=None):
    str_act = str_act.lower()
    assert str_act in ['relu', 'leaky_relu', 'elu', 'sigmoid'], 'Activation function not supported'

    if str_act == 'relu':
        return nn.ReLU()
    elif str_act == 'leaky_relu':
        param = 0.01 if param is None else param
        return nn.LeakyReLU(negative_slope=param) # negative_slope*x for x < 0
    elif str_act == 'elu':
        param = 1.0 if param is None else param
        return nn.ELU(alpha=param) # alpha*(exp(x)-1) for x < 0
    elif str_act == 'sigmoid':
        return nn.Sigmoid()


def get_optimizer(str_opt, model, **kwargs):
    str_opt = str_opt.lower()
    assert str_opt in ['sgd', 'adam', 'adamw'], 'Optimizer not supported'

    if str_opt == 'sgd':
        default_kwargs = {'lr':0.001, 'weight_decay':0, 'momentum':0.95}
        kwargs = {**default_kwargs, **kwargs} # update default kwargs if necessary
        return torch.optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'], momentum=kwargs['momentum'])
    
    elif str_opt == 'adam':
        default_kwargs = {'lr':0.001, 'weight_decay':0}
        kwargs = {**default_kwargs, **kwargs}
        return torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])
    
    elif str_opt == 'adamw':
        default_kwargs = {'lr':0.001, 'weight_decay':0}
        kwargs = {**default_kwargs, **kwargs}
        return torch.optim.AdamW(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])


def get_scheduler(str_sch, optimizer, **kwargs):
    str_sch = str_sch.lower()
    assert str_sch in ['plateau', 'cos', 'exp'], 'Activation function not supported'

    if str_sch == 'plateau': # lr = lr * factor if loss does not decrease for patience epochs on validation set
        default_kwargs = {'factor':0.1, 'patience':5, 'min_lr':1e-5}
        kwargs = {**default_kwargs, **kwargs}
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=kwargs['factor'], patience=kwargs['patience'], min_lr=kwargs['min_lr'], verbose=True)

    elif str_sch == 'cos':
        default_kwargs = {'T_0':10, 'T_mult':2}
        kwargs = {**default_kwargs, **kwargs}
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=kwargs['T_0'], T_mult=kwargs['T_mult'])

    elif str_sch == 'exp': # lr = lr * gamma**epoch
        default_kwargs = {'gamma':0.99}
        kwargs = {**default_kwargs, **kwargs}
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs['gamma'])

    return scheduler


from lib.PLE import compute_bins
# construct dataloader from inputted numpy arrays
def build_dataset(x, y, batch_size_train, batch_size_val, shuffle_train=True, ratio=[0.7, 0.2, 0.1], n_bins=None, seed=0):
    # the shape of x: num_atm * num_vza * num_LST * num_emi * num_input
    # the shape of y: num_atm * num_vza * num_LST * num_emi * num_output
    num_atm, num_input, num_output = x.shape[0], x.shape[-1], y.shape[-1]

    # random split the dataset based on atmospheric profiles
    np.random.seed(seed)
    array_atm = np.arange(num_atm)
    np.random.shuffle(array_atm)
    ratio_train, ratio_val = ratio[0], ratio[1]
    idx_train = array_atm[:int(num_atm*ratio_train)]
    idx_val = array_atm[int(num_atm*ratio_train):int(num_atm*(ratio_train+ratio_val))]
    idx_test = array_atm[int(num_atm*(ratio_train+ratio_val)):]

    x_train, x_val, x_test = x[idx_train].reshape(-1, num_input), x[idx_val].reshape(-1, num_input), x[idx_test].reshape(-1, num_input)
    y_train, y_val, y_test = y[idx_train].reshape(-1, num_output), y[idx_val].reshape(-1, num_output), y[idx_test].reshape(-1, num_output)
    
    if n_bins is not None:
        bin_edges = compute_bins(torch.tensor(x_train, dtype=torch.float32), n_bins=n_bins)
        scaler = None
    else:
        # normalize the input features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        bin_edges = None

    # create tensor datasets
    dataset_train = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataset_val = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    dataset_test = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # create dataloaders
    dataloader_train =  DataLoader(dataset_train, batch_size=batch_size_train, shuffle=shuffle_train)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_val, shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test, idx_train, idx_val, idx_test, scaler, bin_edges


# define QuantileLoss
# if quantile = 0.5, it is the same as L1Loss
# if quantile > 0.5, greater penalties for overestimation, vice versa
class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.max(self.quantile*error, (self.quantile-1)*error)
        return loss.mean()


# construct custiom loss function
class CustomLoss(nn.Module):
    def __init__(self, w1, w2=1, kind='mse', quantile=0.5):
        super(CustomLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        if kind == 'mse':
            self.loss = nn.MSELoss()
        elif kind == 'mae':
            self.loss = nn.L1Loss()
        elif kind == 'quantile':
            self.loss = QuantileLoss(quantile)

    def forward(self, Lg_pred, Lg_true, Ld_pred, Ld_true):
        loss_Lg = self.loss(Lg_pred, Lg_true)
        loss_Ld = self.loss(Ld_pred, Ld_true)
        loss_all = self.w1*loss_Lg + self.w2*loss_Ld
        return loss_all, loss_Lg, loss_Ld


# init weights for the model
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias.data)


# train and evaluate the model
def train_model(model, dataloader_train, dataloader_val, num_epochs, patience, optimizer, device, criterion, scheduler=None, save_path=None, logger=None, seed=0, d_output=3):
    # set seeds for reproducibility
    set_seed(seed)

    num_train, num_val = len(dataloader_train.dataset), len(dataloader_val.dataset)
    loss_eval_all_min = np.inf
    count_not_decrease = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_train_all, loss_train_Lg, loss_train_Ld = 0, 0, 0
        for x_train, y_train in tqdm(dataloader_train):
            x_train, y_train = x_train.to(device), y_train.to(device)
            Lg_prd, Ld_prd = model(x_train)
            loss_train, loss_Lg, loss_Ld = criterion(Lg_prd, y_train[:,:d_output], Ld_prd, y_train[:,d_output:])
            loss_train_all += loss_train.item() * len(x_train) # mean squared error for the batch -> total sum squared error for the batch
            loss_train_Lg += loss_Lg.item() * len(x_train)
            loss_train_Ld += loss_Ld.item() * len(x_train)

            # Backpropagation and update
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        # calculate mean squared error for the total train dataset
        loss_train_all /= num_train
        loss_train_Lg /= num_train
        loss_train_Ld /= num_train

        # Eval
        model.eval()
        loss_eval_all, loss_eval_Lg, loss_eval_Ld = 0, 0, 0
        with torch.no_grad():
            for x_eval, y_eval in dataloader_val:
                x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                Lg_prd, Ld_prd = model(x_eval)
                loss_eval, loss_Lg, loss_Ld = criterion(Lg_prd, y_eval[:,:d_output], Ld_prd, y_eval[:,d_output:])
                loss_eval_all += loss_eval.item() * len(x_eval)
                loss_eval_Lg += loss_Lg.item() * len(x_eval)
                loss_eval_Ld += loss_Ld.item() * len(x_eval)
        
        # calculate mean squared error for the total validation dataset
        loss_eval_all /= num_val
        loss_eval_Lg /= num_val
        loss_eval_Ld /= num_val

        # Update learning rate per epoch
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_eval_all)
            else:
                scheduler.step()

        # Print
        print_string = f'Epoch [{epoch+1}/{num_epochs}], train:{np.sqrt(loss_train_all):.4f}, {np.sqrt(loss_train_Lg):.4f}, {np.sqrt(loss_train_Ld):.4f} | val:{np.sqrt(loss_eval_all):.4f}, {np.sqrt(loss_eval_Lg):.4f}, {np.sqrt(loss_eval_Ld):.4f}'
        print(print_string)
        if logger is not None:
            logger.info(print_string)

        # Save model and early stopping
        if loss_eval_all < loss_eval_all_min:
            count_not_decrease = 0
            loss_eval_all_min = loss_eval_all
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f'Model saved')
        else:
            count_not_decrease += 1
            if count_not_decrease >= patience:
                print_string = f'Early stopping, best validation loss:{np.sqrt(loss_eval_all_min):.4f}'
                print(print_string)
                if logger is not None:
                    logger.info(print_string)
                break


def test_model(model, dataloader_test, device):
    y_test_all, y_test_pred_all = [], []
    model.eval()
    with torch.no_grad():
        for x_test, y_test in tqdm(dataloader_test):
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_test_pred = model(x_test)
            y_test_all.append(y_test.detach().cpu().numpy())
            y_test_pred_all.append(y_test_pred.detach().cpu().numpy())
    y_test_all = np.concatenate(y_test_all, axis=0)
    y_test_pred_all = np.concatenate(y_test_pred_all, axis=0)
    return y_test_all, y_test_pred_all


'''
def train_model(model, dataloader_train, dataloader_val, num_epochs, patience, optimizer, device, criterion=nn.MSELoss(), save_path=None, logger=None, seed=0):
    # set seeds for reproducibility
    set_seed(seed)

    num_train, num_val = len(dataloader_train.dataset), len(dataloader_val.dataset)

    loss_eval_avg_min = np.inf
    count_not_decrease = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_train_avg = 0
        for x_train, y_train in tqdm(dataloader_train):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train_pred = model(x_train)
            loss_train = criterion(y_train_pred, y_train)
            loss_train_avg += loss_train.item() * len(x_train) # mean squared error for the batch -> total sum squared error for the batch
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        loss_train_avg /= num_train # mean squared error for the total train dataset

        # Eval
        model.eval()
        loss_eval_avg = 0
        with torch.no_grad():
            for x_eval, y_eval in dataloader_val:
                x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                y_eval_pred = model(x_eval)
                loss_eval = criterion(y_eval_pred, y_eval)
                loss_eval_avg += loss_eval.item() * len(x_eval)
        loss_eval_avg /= num_val

        # Print
        print_string = f'Epoch [{epoch+1}/{num_epochs}], Loss_train: {np.sqrt(loss_train_avg):.4f}, Loss_val: {np.sqrt(loss_eval_avg):.4f}'
        print(print_string)
        if logger is not None:
            logger.info(print_string)

        # Save model and early stopping
        if loss_eval_avg < loss_eval_avg_min:
            count_not_decrease = 0
            loss_eval_avg_min = loss_eval_avg
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f'Model saved')
        else:
            count_not_decrease += 1
            if count_not_decrease >= patience:
                print_string = f'Early stopping, best validation loss:{np.sqrt(loss_eval_avg_min):.4f}'
                print(print_string)
                if logger is not None:
                    logger.info(print_string)
                break
'''
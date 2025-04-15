import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def extract_csv(file_path, col_names:list, col_names_rename:list=None, type = 'csv'):
    if type == 'csv':
        df = pd.read_csv(file_path)
    elif type == 'excel':
        df = pd.read_excel(file_path)
    
    df = df[col_names]

    if col_names_rename is not None:
        assert len(col_names) == len(col_names_rename), "col_names and col_names_rename must have the same length"
        dicts = {}
        for i in range(len(col_names)):
            dicts[col_names[i]] = col_names_rename[i]
        return df.rename(columns=dicts)
    else:
        return df


# applying the 3-sigma rule to remove outliers
def outlier_removal(x: np.ndarray, multiplier: float=3.0):
    # remove outliers in sequence x
    x_med = np.median(x)
    s = 1.4826 * np.median(np.abs(x-x_med))
    idx_valid = np.where(np.abs(x-x_med) < multiplier*s)[0]
    return idx_valid


def extract_bit(number, length, bit_start, bit_end=None):
    # convert int to binary string, then extract bit at specific loc
    # original bit is from right to left: length-1, length-2, ..., 1, 0
    # bit_start is larger than or equal to bit_end
    # https://spatialthoughts.com/2021/08/19/qa-bands-bitmasks-gee/
    if length == 8:
        bit_str = f'{number:08b}'
    elif length == 16:
        bit_str = f'{number:016b}'

    bit_end = bit_start if bit_end is None else bit_end
    bit_extracted = bit_str[length-bit_start-1: length-bit_end]
    return int(bit_extracted)

extract_bit_vec = np.vectorize(extract_bit) # this will return an array even if only one element is passed


## -------------------------------------------------------------------
## functions used in T-based validation
import torch
from lib.models import *
from lib.tes import *
tes = TES_calculator(num_bands=3)
lamda_c = np.array([8.29587257, 8.81448846, 9.21205838, 10.49186976, 12.08465478])

# DL-SW-TES
# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler_w_tpw = pickle.load(open('result/scaler/scaler_3b.pkl','rb'))
params_model = {'d_in': 5, 'd_out': 3, 'd_block':128, 'd_hidden':64, 'n_block_l':3, 'n_block_h':2, 'act':'sigmoid', 'skip':True}
model_w_tpw = Res_MLP(**params_model).to(device)
weight_path = 'model/res_mlp/model_w_tpw.pt'
model_w_tpw.load_state_dict(torch.load(weight_path))

scaler_wo_tpw = pickle.load(open('result/scaler/scaler_3b_wo_tpw.pkl','rb'))
params_model = {'d_in': 4, 'd_out': 3, 'd_block':128, 'd_hidden':64, 'n_block_l':3, 'n_block_h':2, 'act':'sigmoid', 'skip':True}
model_wo_tpw = Res_MLP(**params_model).to(device)
weight_path = 'model/res_mlp/model_wo_tpw.pt'
model_wo_tpw.load_state_dict(torch.load(weight_path))

def DL_SW_TES(L2, L4, L5, vza, tpw=None):
    dL = L4 - L5
    sec_vza = 1 / np.cos(np.deg2rad(vza))
    if tpw is None:
        x = np.concatenate([L2[:,None], L4[:,None], dL[:,None], sec_vza[:,None]], axis=1)
        x_norm = scaler_wo_tpw.transform(x)
        model = model_wo_tpw
    else:
        x = np.concatenate([L2[:,None], L4[:,None], dL[:,None], sec_vza[:,None], tpw[:,None]], axis=1)
        x_norm = scaler_w_tpw.transform(x)
        model = model_w_tpw

    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)
        results = model(x_tensor)
        Lg_dl, Ld_dl = results[0].cpu().numpy(), results[1].cpu().numpy()

    num_samples = len(Lg_dl)
    lst_dl, emi_dl, qa_dl = np.zeros(num_samples), np.zeros((num_samples,3)), np.zeros(num_samples)
    for i in tqdm(range(num_samples)):
        lst_dl[i], emi_dl[i,:], qa_dl[i], _ = tes(Lg_dl[i], Ld_dl[i])
    
    return lst_dl, emi_dl, qa_dl
    

# load coefs and LUT of SW-TES
coefs_sw_tes = np.load('result/sw_tes/coefs_sw_tes.npy')
Ld_modtran = np.load('result/sw_tes/Ld_modtran.npy')
Ld_LUT = np.load('result/sw_tes/Ld_LUT.npy')
def SW_TES(L2, L4, L5, lat, month):
    T2 = Inverse_Planck_law(L2, lamda_c[1])
    T4 = Inverse_Planck_law(L4, lamda_c[3])
    T5 = Inverse_Planck_law(L5, lamda_c[4])

    d_T2_T5 = T2 - T5
    d_T2_T5_2 = d_T2_T5**2
    d_T4_T5 = T4 - T5
    d_T4_T5_2 = d_T4_T5**2
    d_T5_T4 = T5 - T4
    d_T5_T4_2 = d_T5_T4**2

    inputs_T2 = np.concatenate((np.ones_like(T2[:,None]), T2[:,None], d_T2_T5[:,None], d_T2_T5_2[:,None]), axis=1)
    inputs_T4 = np.concatenate((np.ones_like(T4[:,None]), T4[:,None], d_T4_T5[:,None], d_T4_T5_2[:,None]), axis=1)
    inputs_T5 = np.concatenate((np.ones_like(T5[:,None]), T5[:,None], d_T5_T4[:,None], d_T5_T4_2[:,None]), axis=1)

    Tg2 = inputs_T2 @ coefs_sw_tes[0]
    Tg4 = inputs_T4 @ coefs_sw_tes[1]
    Tg5 = inputs_T5 @ coefs_sw_tes[2]

    Lg_zheng = np.zeros((len(L2), 3))
    Lg_zheng[:, 0] = Planck_law(Tg2, lamda_c[1])
    Lg_zheng[:, 1] = Planck_law(Tg4, lamda_c[3])
    Lg_zheng[:, 2] = Planck_law(Tg5, lamda_c[4])

    # LUT for Ld
    row_in_lut = np.round((80-lat)/10).astype(int)
    row_in_lut[row_in_lut==-1] = 0
    row_in_lut[row_in_lut==17] = 16
    col_in_lut = ((month-1) // 2).astype(int)
    idx = Ld_LUT[row_in_lut, col_in_lut]
    Ld_zheng = Ld_modtran[idx][:,[1,3,4]]

    num_samples = len(Lg_zheng)
    lst_zheng, emi_zheng, qa_zheng = np.zeros(num_samples), np.zeros((num_samples,3)), np.zeros(num_samples)
    for i in tqdm(range(num_samples)):
        lst_zheng[i], emi_zheng[i,:], qa_zheng[i], _ = tes(Lg_zheng[i], Ld_zheng[i])

    return lst_zheng, emi_zheng, qa_zheng


# SWDTES
swdtes_calculator = SWDTES_calculator(sensor='ecostress', n_bands=3)
def SWDTES(L2, L4, L5, tpw, t1=0.1):
    T2 = Inverse_Planck_law(L2, lamda_c[1])
    T4 = Inverse_Planck_law(L4, lamda_c[3])
    T5 = Inverse_Planck_law(L5, lamda_c[4])

    num_samples = len(L2)
    lst_swdtes, emi_swdtes, qa_swdtes = np.zeros(num_samples), np.zeros((num_samples,3)), np.zeros(num_samples)
    for i in tqdm(range(num_samples)):
        lst_swdtes[i], emi_swdtes[i,:], qa_swdtes[i] = swdtes_calculator(T2[i], T4[i], T5[i], tpw[i], t1=t1)
    return lst_swdtes, emi_swdtes, qa_swdtes



## -------------------------------------------------------------------
## functions used in cross-validation
import os
import requests
from zipfile import ZipFile
import h5py

def Read_ECOSTRESS(file_geo=None, file_rad=None, file_lst=None):

    lat, lon, vza, rad2, rad4, rad5, lst, pwv, qc, land_fraction = None, None, None, None, None, None, None, None, None, None

    if file_geo is not None:
        f_geo = h5py.File(file_geo, 'r')
        lat, lon, vza = f_geo['Geolocation']['latitude'][:], f_geo['Geolocation']['longitude'][:], f_geo['Geolocation']['view_zenith'][:]
        land_fraction = f_geo['Geolocation']['land_fraction'][:]
        f_geo.close()
    
    if file_rad is not None:
        f_rad = h5py.File(file_rad, 'r')
        rad2, rad4, rad5 = f_rad['Radiance']['radiance_2'][:], f_rad['Radiance']['radiance_4'][:], f_rad['Radiance']['radiance_5'][:]
        f_rad.close()
    
    if file_lst is not None:
        f_lst = h5py.File(file_lst, 'r')
        # for LST product, note the scale_factor and add_offset are set for LST and PWV
        lst, pwv, qc = f_lst['SDS']['LST'][:], f_lst['SDS']['PWV'][:], f_lst['SDS']['QC'][:]
        scale_lst, offset_lst = f_lst['SDS']['LST'].attrs['scale_factor'][0], f_lst['SDS']['LST'].attrs['add_offset'][0]
        scale_pwv, offset_pwv = f_lst['SDS']['PWV'].attrs['scale_factor'][0], f_lst['SDS']['PWV'].attrs['add_offset'][0]
        lst = lst * scale_lst + offset_lst
        pwv = pwv * scale_pwv + offset_pwv
        f_lst.close()

    return lat, lon, vza, rad2, rad4, rad5, lst, pwv, qc, land_fraction

def Download_imgs(args):

    img, region, save_path, output_name, scale, unmask = args
    
    output_path = os.path.join(save_path, output_name) + '.zip'
    
    if unmask is not None:
        img = img.unmask(unmask)

    # get the download URL from GEE, then # download file from the url
    try:
        url = img.getDownloadURL({
            'name': output_name,
            'region': region,
            'scale': scale,
            'format': 'ZIPPED_GEO_TIFF',
            'crs': 'EPSG:4326'})
        response = requests.get(url)

    except Exception as e:
        print("Failed to download " + output_name)
        print(e)
        return args

    with open(output_path, 'wb') as fd:
        fd.write(response.content)
        
    with ZipFile(output_path, 'r') as zipObj:
        zipObj.extractall(save_path)
            
    os.remove(output_path)
            
    # print(output_path + " has been downloaded successfully!")


import rasterio
def read_geo(file):
    dataset = rasterio.open(file)
    data = dataset.read(1)
    
    trans = dataset.get_transform()
    x_ul, x_res, y_ul, y_res = trans[0], trans[1], trans[3], trans[5]
    y_num, x_num = data.shape
    x = np.linspace(x_ul + x_res/2, x_ul + x_res/2 + x_res * (x_num-1), x_num)
    y = np.linspace(y_ul + y_res/2, y_ul + y_res/2 + y_res * (y_num-1), y_num)
    lon, lat = np.meshgrid(x, y)

    return data, lon, lat


def encode_igbp(array):
    # forest
    array[array<=5] = 100
    # shurbland
    array[(array>5) & (array<8)] = 200
    # savanna
    array[(array>7) & (array<10)] = 300
    # grassland
    array[array==10] = 400
    # wetland
    array[array==11] = 500
    # cropland
    array[(array==12) | (array==14)] = 600
    # urban
    array[array==13] = 700
    # snow
    array[array==15] = 800
    # barren
    array[array==16] = 900
    # water
    array[array==17] = 1000
    return array
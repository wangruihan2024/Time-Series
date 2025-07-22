import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
from tqdm import tqdm
import joblib

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Autoformer')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='Close')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--train_path', type=str, default=None, help='Path to training CSV (for test-time history lookup)')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=True)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25)

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

    # model define
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=96)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False)

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--jitter', default=False, action="store_true")
    parser.add_argument('--scaling', default=False, action="store_true")
    parser.add_argument('--permutation', default=False, action="store_true")
    parser.add_argument('--randompermutation', default=False, action="store_true")
    parser.add_argument('--magwarp', default=False, action="store_true")
    parser.add_argument('--timewarp', default=False, action="store_true")
    parser.add_argument('--windowslice', default=False, action="store_true")
    parser.add_argument('--windowwarp', default=False, action="store_true")
    parser.add_argument('--rotation', default=False, action="store_true")
    parser.add_argument('--spawner', default=False, action="store_true")
    parser.add_argument('--dtwwarp', default=False, action="store_true")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true")
    parser.add_argument('--wdba', default=False, action="store_true")
    parser.add_argument('--discdtw', default=False, action="store_true")
    parser.add_argument('--discsdtw', default=False, action="store_true")
    parser.add_argument('--extra_tag', type=str, default="")
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--predict_output', type=str, default='TimeXer.csv')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl', help='path to saved scaler.pkl')
    parser.add_argument('--cache_dataset_path', type=str, default='./cache/traffic_preprocessed.pkl',
                    help='Path to save/load cached test dataset')
    

    # TimesNet
    parser.add_argument('--patch_len', type=int, default=16)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
                args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
                args.factor, args.embed, args.distil, args.des, ii)

            print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print('>>>>>>>start testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
            args.factor, args.embed, args.distil, args.des, ii)

        if args.do_predict:
            import pandas as pd
            import joblib
            from tqdm import tqdm
            from torch.utils.data import DataLoader
            from data_provider.data_loader import Dataset_Traffic_QPredict

            print('>>>>>>>running prediction on test set<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            model = exp.model
            model.eval()

            # 读取数据
            test_df = pd.read_csv(os.path.join(args.root_path, args.data_path))
            train_df = pd.read_csv(os.path.join(args.root_path, args.train_path))
            test_df.columns = test_df.columns.str.strip().str.replace("'", "")
            train_df.columns = train_df.columns.str.strip().str.replace("'", "")
            test_df['date'] = pd.to_datetime(test_df['date'])
            train_df['date'] = pd.to_datetime(train_df['date'])

            # 构建 Dataset 和 DataLoader
            cache_path = args.cache_dataset_path
            os.makedirs('./cache', exist_ok=True)

            if os.path.exists(cache_path):
                print(f'Loading cached dataset from {cache_path}')
                test_dataset = joblib.load(cache_path)
                print(f'Loading Finished')
            else:
                print(f'Building dataset and caching to {cache_path}')
                test_dataset = Dataset_Traffic_QPredict(
                    args=args,
                    root_path=args.root_path,
                    size=(args.seq_len, args.label_len, args.pred_len),
                    data_path=args.data_path,
                    train_df=train_df,
                    scaler_path=args.scaler_path,
                )
                joblib.dump(test_dataset, cache_path)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False
            )
            
            # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            # args.task_name,
            # args.model_id,
            # args.model,
            # args.data,
            # args.features,
            # args.seq_len,
            # args.label_len,
            # args.pred_len,
            # args.d_model,
            # args.n_heads,
            # args.e_layers,
            # args.d_layers,
            # args.d_ff,
            # args.expand,
            # args.d_conv,
            # args.factor,
            # args.embed,
            # args.distil,
            # args.des, ii)
            
            exp.test2(setting=setting, test=1, test_loader=test_loader, test_data=test_dataset)
            print(f'>>>>>>>Prediction results saved to output/predictions <<<<<<<<')
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)

        if args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
    
    
    
    
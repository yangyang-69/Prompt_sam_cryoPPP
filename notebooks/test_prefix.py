
from dataset import *
from torch.utils.data import DataLoader
from SAM_adapter_conf.SAM_adapter_utils import *
import function_token

args = SAM_adapter_cfg.parse_args()
if args.dataset == 'refuge' or args.dataset == 'refuge2':
    args.data_path = '../dataset'

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

args.path_helper = checkpoint['path_helper']
logger = create_logger(args.path_helper['log_path'])
print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('../logs', args.exp_name)
# args.path_helper = set_log_dir('../54321test', args.exp_name)
# args.path_helper = set_log_dir('../whether_finetune', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.out_size, args.out_size)),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.out_size, args.out_size)),

])
'''data end'''
if args.dataset == 'CryoPPP':
    # turkey_valid_dataset = CryopppDataset(args, args.data_path, transform=transform_test,
    #                                       transform_msk=transform_test_seg, mode='test', prompt=args.prompt_approach)

    turkey_valid_dataset = CryopppDataset(args, args.data_path, transform=transform_test,
                                          transform_msk=transform_test_seg, mode='test', prompt=args.prompt_approach)

    nice_test_loader = DataLoader(turkey_valid_dataset, batch_size=args.b, shuffle=False, num_workers=8,
                                  pin_memory=True)
    '''end'''


elif args.dataset == 'decathlon':
    nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

if args.mod == 'sam_token_prompt':
    net.eval()
    tol, (eiou, edice) = function_token.Test_sam(args, nice_test_loader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')

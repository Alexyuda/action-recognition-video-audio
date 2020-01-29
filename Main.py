import torch
from opts import parser
import os
import pickle
from DateSet import MYUCF101

def load_data_set(args, train):
    # data loaders
    meta_data_train_str = os.path.join(args.dataset_dir,
                                       f"meta_data_train_{train}_fold_{args.fold}_frames_"
                                       f"{args.frames_per_clip}_skip_{args.step_between_clips}.pickle")
    if os.path.exists(meta_data_train_str):
        with open(meta_data_train_str, 'rb') as f:
            meta_data = pickle.load(f)
    else:
        meta_data = None

    trainset = MYUCF101(root=args.dataset_dir,
                        annotation_path=args.train_test_split_dir,
                        frames_per_clip=args.frames_per_clip, step_between_clips=args.step_between_clips,
                        fold=args.fold, train=train, _precomputed_metadata=meta_data, num_workers=0)
    return trainset

def main():
    args = parser.parse_args()

    # Constant seed
    torch.manual_seed(0)
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = load_data_set(args=args, train=True)
    testset = load_data_set(args=args, train=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)




if __name__ == '__main__':
    main()

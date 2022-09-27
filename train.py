import argparse
import bisect
import datetime
import functools
import json
import os
import time
import itertools
import torch
import torch.utils.data
import numpy as np
from sketchgraphs.data import flat_array
from data import CADDataset, collate_fn, SubsetSampler
from TrainingHarness import *
import torch.multiprocessing as mp
import torch.distributed as dist
from model_embedding import *
from param_model import ParamModel
from model import CADModel
                                       
def get_argsparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', "-d",required=True,
                        help='Path to training dataset')\
    
    parser.add_argument('--train_file', required=True,
                        help='The filename of the training data')\

    parser.add_argument('--experiment', "-e",required=True,
                        help='Path to experiment directory')\

    parser.add_argument('--continue', "-c",required=False,
                        help='continue from specified epoch')

    parser.add_argument('--visualization', "-v",required=False,default=False,\
                        help='continue from specified epoch')
    return parser

def load_experiment_specification(experiment_directory):
    filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file ".format(experiment_directory)
            + '"specs.json"'
        )

    return json.load(open(filename))


def make_train_dataset(args,file):
    filename = os.path.join(args["dataset_dir"],file)
    GT_sketch_data = flat_array.load_dictionary_flat(filename)
    GT_sketch = GT_sketch_data["sequences"]
    if not os.path.isfile(filename):
        raise Exception(
            "The data directory ({}) does not include validation data file "
            + '({})'.format(args["dataset_dir"],file)
        )
    
    if os.path.exists(filename[:-4] + "_filter.npy"):
        correspondance_data = np.load(filename[:-4] + "_filter.npy")
        return {
            "correspond": correspondance_data,
            "GT_sketch" : GT_sketch
        }
    else:
        return {
            "GT_sketch" : GT_sketch
        }

def initialize_datasets(rank,world_size,args,specs):

    #load the raw proprocessed_data
    print("Initializing dataset")
    train_data =  make_train_dataset(args,args["train_file"])
    ds_train = CADDataset( train_data,specs,False,rank)
    train_sampler = SubsetSampler(rank,world_size,False, specs,specs["seed"])


    validation_data =  make_train_dataset(args,args["train_file"])
    ds_validation = CADDataset( validation_data,specs,True,rank)
    validation_sampler = SubsetSampler(rank,world_size,True, specs,specs["seed"])

    batch_size = specs['batch_size'] 
    validation_batch_size = specs["validation_batch_size"]
    num_workers = specs['num_workers']

    print("Initialize dataloader")
    dataloader_train = torch.utils.data.DataLoader(
        ds_train,
        collate_fn = collate_fn,
        batch_size = batch_size,
        num_workers=num_workers,
        sampler = train_sampler)

    print("training with {} sketches".format(len(train_sampler)))

    dataloader_validation = torch.utils.data.DataLoader(
        ds_validation,
        collate_fn =  collate_fn,
        batch_size = validation_batch_size,
        num_workers=num_workers,
        sampler = validation_sampler)
    print("validation with {} sketches".format(len(validation_sampler)))
    return dataloader_train, train_sampler, dataloader_validation, validation_sampler



def initialize_optimizer(specs, model, embedding):
    def Add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = specs["gradient_clip_value"]
        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if specs["clip_gradient"] else optim

    params = []
    for key, value in model.named_parameters(recurse=True):   ##Here need to alternate the training parameter length maybe?
        if not value.requires_grad:
            continue
        lr = specs["base_lr"]
        params += [{"params": [value], "lr": lr}]
    
    if embedding != None:
        for key, value in embedding.named_parameters(recurse=True):   ##Here need to alternate the training parameter length maybe?
            if not value.requires_grad:
                continue
            lr = specs["base_lr"]
            params += [{"params": [value], "lr": lr}]

    optimizer_type = specs["optimizer_type"]
    if optimizer_type == "ADAM":
        optimizer = Add_full_model_gradient_clipping(torch.optim.Adam)(
            params, specs["base_lr"],
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")

    return optimizer


def initialize_model(specs,rank):
    model = CADModel(specs["NetworkSpecs"],specs["EmbeddingStructure"],rank, specs["seed"], specs)
    return model

def initialize_param_model(specs,args,rank):
    model = ParamModel(specs["param_model"]["arch"],args,specs["param_model"]["embeded"],specs,specs["EmbeddingStructure"]["type_schema"],rank)
    return model

def initialize_embedding(specs,rank):
    embedding = CADEmbedding(specs,rank)         
    return embedding
 
def train(rank,world_size, args,specs, output_dir):
    #####set up for distributed training

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Running on rank {rank}.")

    print('Loading datasets')
    print(specs['description'],specs["exp_name"])
    dl_train, train_sampler, dl_validation, validation_sampler= initialize_datasets(rank,world_size,
        args, specs)


    print("config the saving")
    total_num_epoch = specs["NumEpochs"]
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            total_num_epoch + 1,
            specs["SnapshotFrequency"],
        )
    )


    harness = TrainingHarness(dl_train,dl_validation,output_dir,args,specs, train_sampler, validation_sampler ,checkpoints,rank)
    if args["visualization"]:
        harness.sketch_visualization()
    else:
        while harness.epoch <= specs["NumEpochs"]:
            harness.single_step()




def run(args, specs):
    """Runs the entire training process according to the given configuration.
    """
    # Set seeds
    np.random.seed(specs['seed'])
    torch.manual_seed(specs['seed'])

    print("create directory to save the training data") # for starting a brandnew training from skratch
    result_path = os.path.join(args["experiment"],"result")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    param_path = os.path.join(result_path,"network_param")
    optim_path = os.path.join(result_path,"optimizer")
    visulization_path = os.path.join(result_path,"sketch_visualization")
    paths = [param_path,optim_path,visulization_path]
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)

    
    if not args["visualization"] and specs["param_code_flag"]:
        raise NotImplementedError("Training with param_code mode")

    print('Starting training.')
    start_time = time.perf_counter()
    if args["visualization"]:
        world_size = 1
    else:
        world_size = torch.cuda.device_count()
    print("training with {} GPU(s)".format(world_size))
    mp.spawn(train, args= (world_size, args,specs,result_path,),nprocs=world_size,join=True)

    end_time = time.perf_counter()
    print(f'Done training. Total time: {datetime.timedelta(seconds=end_time - start_time)}.')



def main():
    """Default main function."""
    parser = get_argsparser()
    args = parser.parse_args()

    specs = load_experiment_specification(args.experiment)
    run(vars(args),specs)

if __name__ == '__main__':
    main()



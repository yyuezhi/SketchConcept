import os
import torch
import numpy as np
from sketchgraphs.data import flat_array, sequence
OPTIMIZER_DIR = "optimizer"
MODEL_PARAM_DIR = "network_param"
SKETCH_VISUALIZATION_DIR = "sketch_visualization"

####Saving and loading ###########
def save_model(self,filename, epoch):
    torch.save(
    {"epoch": epoch, "model_state_dict": self.model.state_dict(),
        "embedding_state_dict" : self.cad_embedding.state_dict()},
    os.path.join(self.output_dir,MODEL_PARAM_DIR, filename),
    )

def save_optimizer(self,filename,epoch):
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
        os.path.join(self.output_dir,OPTIMIZER_DIR, filename),
    )

def save_checkpoints(self,epoch):
    self.save_model( "model_"+str(epoch) + ".pth", epoch)
    self.save_optimizer("optim_"+ str(epoch) +".pth", epoch)

def load_optimizer(self,filename):
    full_filename = os.path.join( self.output_dir, OPTIMIZER_DIR, filename)

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
    data = torch.load(full_filename,map_location=map_location)
    
    self.optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["epoch"]

def load_model_parameters(self,filename):

    full_filename = os.path.join(self.output_dir, MODEL_PARAM_DIR, filename)
    if not os.path.isfile(full_filename):
        raise Exception('model state dict "{}" does not exist'.format(full_filename))

    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
    data = torch.load(full_filename,map_location=map_location)

    self.model.load_state_dict(data["model_state_dict"])

    self.cad_embedding.load_state_dict(data["embedding_state_dict"])

    return data["epoch"]


def save_data(self):
    """This function is called at the end of each epoch."""
    if self.epoch in self.checkpoints or self.epoch <= 10:
        self.save_checkpoints(self.epoch)
    



####The below two is for validation logging only
def save_epoch_accuracy_log(self,type_accuracy,refer_accuracy, non_type_accuracy):
    self.writer.add_scalar("Accuracy/epoch_type_accuracy",type_accuracy,self.epoch)
    self.writer.add_scalar("Accuracy/epoch_refer_accuracy",refer_accuracy,self.epoch)
    self.writer.add_scalar("Accuracy/epoch_non_type_accuracy",non_type_accuracy,self.epoch)
    self.writer.flush()


def save_log(self,epoch_loss,
                epoch_entity_loss,
                epoch_reference_loss,
                epoch_non_type_loss,
                epoch_reg_loss,
            epoch_entity_type_loss,
            epoch_validity_loss,
            epoch_cost_matrix_reference_loss,
            epoch_commitment_loss):
    factor =  self.num_batch_per_train
    self.writer.add_scalar("Loss/epoch_loss",epoch_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_entity_loss",epoch_entity_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_reference_loss",epoch_reference_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_non_type_loss",epoch_non_type_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_reg_loss",epoch_reg_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_entity_type_loss",epoch_entity_type_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_validity_loss",epoch_validity_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_cost_matrix_reference_loss",epoch_cost_matrix_reference_loss/factor,self.epoch)
    self.writer.add_scalar("Loss/epoch_commitment_loss",epoch_commitment_loss/factor,self.epoch)
    self.writer.flush()

####saving done#####
###saving and loading above##########
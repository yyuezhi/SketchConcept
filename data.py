from sketchgraphs.data.sketch import Sketch
import torch
import numpy as np
import math
from sketchgraphs.data._constraint import LocalReferenceParameter, ConstraintType
from sketchgraphs.data._entity import Arc , Line, Circle, Point
from model_embedding import string_to_float
from sketchgraphs.data._entity import EntityType


def collate_fn(batch):
    sketch_GT,  sketch_length,encoding,GT_label = zip(*batch)
    type_idx, refer_idx, refer_posi, primitive_param_posi,constraint_param_posi,primitive_GT,constraint_GT,  primitive_length,constraint_length = zip(*encoding)
    type_GT,type_mask, refer_orig_GT,refer_seq_GT,refer_dest_GT,batch_prim_param_GT,batch_const_param_GT,batch_const_param_type_GT = zip(*GT_label)
    sketch_length_csum = np.cumsum(np.array([0] + list(sketch_length)))[:-1]
    type_batch_idx = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(type_idx)],dim=0)
    type_token_idx = torch.cat([torch.arange(i.shape[0]) for idx,i in enumerate(type_idx)],dim=0)
    refer_batch_idx = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(refer_posi)],dim=0)
    primitive_param_batch = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(primitive_param_posi)],dim=0)
    constraint_param_batch = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(constraint_param_posi)],dim=0)

    refer_batch_GT = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(refer_seq_GT)],dim=0)
    ## type_idx and refer_idx
    type_idx = torch.cat(type_idx,dim=0)
    refer_idx = torch.cat(refer_idx,dim=0)
    refer_posi = torch.cat(refer_posi,dim=0)

    primitive_param_posi = torch.cat(primitive_param_posi,dim=0)
    constraint_param_posi = torch.cat(constraint_param_posi,dim=0)
    ##GT for loss calculation
    type_batch_GT = torch.cat([torch.ones(i.shape[0],dtype = torch.long) * idx for idx,i in enumerate(type_GT)],dim=0)
    primitive_length = torch.tensor(primitive_length,dtype = torch.int64)
    constraint_length = torch.tensor(constraint_length,dtype = torch.int64)
    type_GT = torch.cat(type_GT,dim=0)
    primitive_GT = torch.cat(primitive_GT,dim=0)
    constraint_GT = torch.cat(constraint_GT,dim=0)
    type_mask = torch.cat(type_mask,dim=0)
    refer_orig_GT = torch.cat([refer_orig_GT[idx] + sum for idx,sum in enumerate(sketch_length_csum)],dim=0)
    refer_seq_GT = torch.cat(refer_seq_GT,dim=0)
    refer_dest_GT = torch.cat([refer_dest_GT[idx] + sum for idx,sum in enumerate(sketch_length_csum)],dim=0)
    batch_prim_param_GT = torch.cat(batch_prim_param_GT,dim=0)
    batch_const_param_GT = torch.cat(batch_const_param_GT,dim=0)
    batch_const_param_type_GT = torch.cat(batch_const_param_type_GT,dim=0)
    return sketch_GT,sketch_length,(type_idx, type_batch_idx,type_token_idx, refer_idx,refer_batch_idx,refer_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT, batch_const_param_GT,batch_const_param_type_GT,primitive_GT, constraint_GT, primitive_length, constraint_length),\
        (type_GT,type_mask, type_batch_GT,refer_batch_GT,refer_seq_GT,refer_dest_GT,refer_orig_GT,batch_prim_param_GT,primitive_GT)



class CADDataset(torch.utils.data.Dataset):
    def __init__(self, data,specs,validation_flag,rank):
        self.GT_sequences = data["GT_sketch"]
        if "correspond" in data.keys():
            if validation_flag:
                self.correspondance = data["correspond"][:specs["test_data"]["dataset_size"]]
            else:
                self.correspondance = data["correspond"][-specs["train_data"]["dataset_size"]:]
            self.length = len(self.correspondance)
            self.filter_flag = True
        else:
            if validation_flag:
                self.GT_sequences = self.GT_sequences[:specs["test_data"]["dataset_size"]]
            else:
                self.GT_sequences = self.GT_sequences[-specs["train_data"]["dataset_size"]:]
            self.length = len(self.GT_sequences)
            self.filter_flag = False

        self.schema_length = len(specs["EmbeddingStructure"]["type_schema"])
        self.refer_dim = 51
        self.num_library = specs["EmbeddingStructure"]["num_library"]
        self.length_quantization = 20
        self.angle_quantization = 30
        self.coordinate_quantization = 80
        self.device =  torch.device(rank if torch.cuda.is_available() else "cpu")
        self.primitive_type_mapping_table = {  #19/20/21 is belong to the start new end tokens 
            EntityType.Line: torch.tensor([0],dtype=torch.long,device=self.device),
            EntityType.Point: torch.tensor([1],dtype=torch.long,device=self.device),
            EntityType.Circle:torch.tensor([2],dtype=torch.long,device=self.device),
            EntityType.Arc: torch.tensor([3],dtype=torch.long,device=self.device),
        }
        self.constraint_type_mapping_table = {
            ConstraintType.Coincident: torch.tensor([4],dtype=torch.long,device=self.device),
            ConstraintType.Distance: torch.tensor([5],dtype=torch.long,device=self.device),
            ConstraintType.Horizontal : torch.tensor([6],dtype=torch.long,device=self.device),
            ConstraintType.Parallel : torch.tensor([7],dtype=torch.long,device=self.device),
            ConstraintType.Vertical : torch.tensor([8],dtype=torch.long,device=self.device),
            ConstraintType.Tangent : torch.tensor([9],dtype=torch.long,device=self.device),
            ConstraintType.Length : torch.tensor([10],dtype=torch.long,device=self.device),
            ConstraintType.Perpendicular : torch.tensor([11],dtype=torch.long,device=self.device),
            ConstraintType.Midpoint : torch.tensor([12],dtype=torch.long,device=self.device),
            ConstraintType.Equal : torch.tensor([13],dtype=torch.long,device=self.device),
            ConstraintType.Diameter : torch.tensor([14],dtype=torch.long,device=self.device),
            ConstraintType.Radius : torch.tensor([15],dtype=torch.long,device=self.device),
            ConstraintType.Angle : torch.tensor([16],dtype=torch.long,device=self.device),
            ConstraintType.Concentric : torch.tensor([17],dtype=torch.long,device=self.device),
            ConstraintType.Normal : torch.tensor([18],dtype=torch.long,device=self.device),
        }

        self.validity_table = {
            ConstraintType.Horizontal:2,
            ConstraintType.Vertical : 2,
            ConstraintType.Midpoint : 3,
        }


    def length_quantitize(self,num): #length range [0,2]
        return int((num%2)/2.00001 * self.length_quantization)

    def angle_quantitize(self,num):
        return int((num%(2*math.pi))/((2*math.pi)+1e-6) * self.angle_quantization)

    def coordinate_quantitize(self,num): #coordinate range [-1,1]
        return int(((num+1)%2)/2.00001 * self.coordinate_quantization)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sequence_idx,enum_idx = idx
        sequence_idx = sequence_idx % self.length
        if self.filter_flag:
            GT_sketch = self.GT_sequences[self.correspondance[sequence_idx]]
        else:
            GT_sketch = self.GT_sequences[sequence_idx]
        sketch_length = len(GT_sketch.entities) + len(GT_sketch.constraints)

        ###extract data from the GT sketch: parameter and trucated things
        #primitive_percentage = random.uniform(0, 1) * 0.5 + 0.5
        batch_primitive_parameter_GT_list = []
        primitive_length = len(GT_sketch.entities)
        constraint_length = len(GT_sketch.constraints)
        for entity_idx, entity_key in enumerate(GT_sketch.entities): 
            entity = GT_sketch.entities[entity_key]
            if type(entity) == Line:
                primitive_parameter = [entity.isConstruction, self.coordinate_quantitize(entity.start_point[0]), self.coordinate_quantitize(entity.start_point[1]),self.coordinate_quantitize(entity.end_point[0]),self.coordinate_quantitize(entity.end_point[1])]
            elif type(entity) == Circle:
                primitive_parameter = [entity.isConstruction, self.coordinate_quantitize(entity.xCenter), self.coordinate_quantitize(entity.yCenter),self.length_quantitize(entity.radius)]
            elif type(entity) == Point:
                primitive_parameter = [entity.isConstruction, self.coordinate_quantitize(entity.x), self.coordinate_quantitize(entity.y)]
            elif type(entity) == Arc:
                primitive_parameter = [entity.isConstruction, self.coordinate_quantitize(entity.xCenter), self.coordinate_quantitize(entity.yCenter), self.length_quantitize(entity.radius),self.angle_quantitize(entity.startParam),self.angle_quantitize(entity.endParam)]
            batch_primitive_parameter_GT_list.append(torch.tensor(primitive_parameter,dtype=torch.int64))

        batch_primitive_parameter_GT = torch.nn.utils.rnn.pad_sequence(batch_primitive_parameter_GT_list,padding_value= - 1)
        batch_prim_param_GT = torch.cat([batch_primitive_parameter_GT,torch.ones(6- batch_primitive_parameter_GT.shape[0],batch_primitive_parameter_GT.shape[1],dtype = torch.int64) * -1 ],dim=0).transpose(0,1)

        batch_constraint_param_GT_list = []
        batch_constraint_param_type_list = []
        for constraint_idx, constraint_key in enumerate(GT_sketch.constraints):
            constraint = GT_sketch.constraints[constraint_key]
            constraint_value_list = [-1]
            constraint_type_list = [-1]
            reference_count = 0
            for i,param in enumerate(constraint.parameters):
                if param.parameterId == 'angle':
                    constraint_value_list.append(self.angle_quantitize(string_to_float(param.expression)))
                    constraint_type_list.append(0)
                elif param.parameterId == 'length':
                    constraint_value_list.append(self.length_quantitize(string_to_float(param.expression)))
                    constraint_type_list.append(1)
                elif isinstance(param, LocalReferenceParameter):
                    reference_count += 1
            if constraint.type in self.validity_table:
                if self.validity_table[constraint.type] == reference_count:
                    constraint_value_list.append(1)
                else:
                    constraint_value_list.append(0)
                constraint_type_list.append(2)
            batch_constraint_param_type_list.append(torch.tensor(constraint_type_list,dtype=torch.int64))
            batch_constraint_param_GT_list.append(torch.tensor(constraint_value_list,dtype=torch.int64))        ### experimental preprocessing boosting
        batch_const_param_GT = torch.nn.utils.rnn.pad_sequence(batch_constraint_param_GT_list,padding_value= - 1)
        batch_const_param_GT = torch.cat([batch_const_param_GT,torch.ones(5- batch_const_param_GT.shape[0],batch_const_param_GT.shape[1],dtype = torch.int64) * -1 ],dim=0).transpose(0,1)

        batch_const_param_type_GT = torch.nn.utils.rnn.pad_sequence(batch_constraint_param_type_list,padding_value= - 1)
        batch_const_param_type_GT = torch.cat([batch_const_param_type_GT,torch.ones(2- batch_const_param_type_GT.shape[0],batch_const_param_type_GT.shape[1],dtype = torch.int64) * -1 ],dim=0).transpose(0,1)

        #### also deal with the type and reference information
        type_idx = []
        primitive_GT = []
        constraint_GT = []
        refer_idx = []
        refer_posi = []
        primitive_param_posi = []
        constraint_param_posi = []
        ####
        type_mask = []
        type_GT = []
        refer_orig_GT = []
        refer_seq_GT = []
        refer_dest_GT = []

        counter = 0
        type_idx.append(self.schema_length)
        counter += 1
        entity_index_dict = {}
        ####New things####
        for entity_idx,entity_key in enumerate(GT_sketch.entities):
            entity_node = GT_sketch.entities[entity_key]
            ttype = self.primitive_type_mapping_table[entity_node.type].item()
            entity_index_dict[entity_key] = entity_idx
            primitive_GT.append(ttype)

            if not (ttype<=3 or ttype in (5,10,16,14,15)):
                type_mask.append(0.0)
            else:
                type_mask.append(1.0)
            type_GT.append(ttype)

            primitive_param_posi.append(counter)

            counter += 1

            ### add type 
            type_idx.extend([ttype])


            ###new token
            type_idx.append(self.schema_length+ 1)
            counter += 1
        ###3New Things

        for constraint_idx,constraint_key in enumerate(GT_sketch.constraints):
            constraint_idx = constraint_idx + len(GT_sketch.entities)
            constraint_node = GT_sketch.constraints[constraint_key]
            ttype = self.constraint_type_mapping_table[constraint_node.type].item()
            if ttype in (5,10,16,14,15):
                constraint_GT.append(ttype)
            else:
                constraint_GT.append(ttype)

            if not (ttype<=3 or ttype in (5,10,16,14,15)):
                type_mask.append(0.0)
            else:
                type_mask.append(1.0)

            type_GT.append(ttype)

            reference_count = 0
            for i,param in enumerate(constraint_node.parameters):  #decode_result [B,S1,L,E]
                if isinstance(param, LocalReferenceParameter):
                    refer_orig_GT.append(constraint_idx)
                    refer_dest_GT.append(entity_index_dict[param.get_referenceMain()])
                    refer_seq_GT.append(reference_count)
                    reference_count += 1

            
            ### add parameter
            constraint_param_posi.append(counter)
            counter += 1

            ###add reference
            reference_count = 0
            for i,param in enumerate(constraint_node.parameters):  #decode_result [B,S1,L,E]
                if isinstance(param, LocalReferenceParameter):
                    refer_posi.append(counter)
                    counter += 1
                    refer_idx.append(entity_index_dict[param.get_referenceMain()])
                    reference_count += 1



            ### add type 
            type_idx.extend([ttype]* (reference_count+1))

            ###new token
            type_idx.append(self.schema_length+ 1)
            counter += 1


        type_idx.pop()
        type_idx.append(self.schema_length+ 2)

        type_idx = torch.tensor(type_idx)
        primitive_GT  = torch.tensor(primitive_GT)
        constraint_GT = torch.tensor(constraint_GT)
        refer_idx = torch.tensor(refer_idx).long()
        refer_posi = torch.tensor(refer_posi).long()
        primitive_param_posi = torch.tensor(primitive_param_posi).long()
        constraint_param_posi = torch.tensor(constraint_param_posi).long()

        type_GT = torch.tensor(type_GT)
        type_mask = torch.tensor(type_mask)
        refer_orig_GT = torch.tensor(refer_orig_GT)
        refer_seq_GT = torch.tensor(refer_seq_GT)
        refer_dest_GT = torch.tensor(refer_dest_GT)

        return GT_sketch, sketch_length, (type_idx, refer_idx, refer_posi, primitive_param_posi,constraint_param_posi,primitive_GT,constraint_GT, primitive_length,constraint_length),\
            (type_GT,type_mask, refer_orig_GT,refer_seq_GT,refer_dest_GT,batch_prim_param_GT,batch_const_param_GT,batch_const_param_type_GT)



class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, rank,world_size,validation,specs,seed):
        np.random.seed(seed)
        print("sampler seed is", seed)
        if validation:
            self.total_data_size = specs["test_data"]["dataset_size"]
        else:
            self.total_data_size = specs["train_data"]["dataset_size"]
        self.indices = list(range(self.total_data_size))
        self.validation = validation
        #adapt for multiprocessing
        self.rank = rank
        self.world_size = world_size
        #drop last!
        self.num_samples = math.ceil(len(self.indices) / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        g = torch.Generator()
        if self.validation:
            g.manual_seed(0)
            idx = list(range(len(self.indices)))
        else:
            g.manual_seed(self.epoch)
            idx = torch.randperm(len(self.indices), generator=g).tolist()
        #Do NOT drop last
        #idx = torch.randperm(len(self.indices), generator=g).tolist()
        padding_size = self.total_size - len(self.indices)
        idx += idx[:padding_size]

        assert len(idx) == self.total_size

        #subsample
        subsample_idx = idx[self.rank:self.total_size:self.world_size]
        indices = [self.indices[i] for i in subsample_idx]

        assert len(indices) == self.num_samples
        zip_indices = zip(indices, subsample_idx)
        return iter(zip_indices)

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self,epoch):
        self.epoch = epoch
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from sketchgraphs.data._entity import EntityType
from sketchgraphs.data._constraint import ConstraintType, LocalReferenceParameter
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from train import initialize_optimizer, initialize_model, initialize_param_model, initialize_embedding


OPTIMIZER_DIR = "optimizer"
MODEL_PARAM_DIR = "network_param"
SKETCH_VISUALIZATION_DIR = "sketch_visualization"

cmd = "nvidia-smi"
class TrainingHarness():
    from validate import validate_one_step,accuracy_evaluation,compare_primitive
    from load_and_save import load_optimizer,load_model_parameters,save_epoch_accuracy_log,save_data,save_log,save_checkpoints,save_optimizer,save_model
    from visualization import sketch_visualization,prediction_to_sketch,initialize_network,library_processing,graph_to_sketch,save_visualize_sketch,constraint_node2graph
    def __init__(self,dl_train,dl_validation,output_dir,args,specs,train_sampler, validation_sampler,checkpoints,rank):
        self.train_sampler = train_sampler
        self.validation_sampler = validation_sampler

        self.model = initialize_model(specs,rank)
        self.model.param_encode_model = initialize_param_model(specs,args,rank)
        self.cad_embedding = initialize_embedding(specs,rank)
        self.cad_embedding =  self.cad_embedding.to(rank)
        self.cad_embedding = DDP(self.cad_embedding,device_ids=[rank])

        self.specs = specs
        #move model to GPU
        print("initialize Distributed Data Parallel")
        self.model = self.model.to(rank)
        self.model = DDP(self.model,device_ids=[rank])
        self.num_library = specs["EmbeddingStructure"]["num_library"]
        self.schema_length = len(specs["EmbeddingStructure"]["type_schema"])

        self.max_detection_query = specs["EmbeddingStructure"]["max_detection_query"]
        self.max_abstruction = specs["EmbeddingStructure"]["max_abstruction_decompose_query"]
        self.rank = rank
        self.dl_train = dl_train
        self.code_regulation = specs["code_regulation"]
        self.dl_validation = dl_validation
        self.output_dir = output_dir
        self.validate_freq = specs["validate_freq"]
        self.data_dir = args["dataset_dir"]
        self.args = args
        self.expname = specs["exp_name"]
        self.num_batch_per_train = len(dl_train)
        self.specs = specs
        self.param_code_encoding_ratio = specs["param_code_encoding_ratio"]

        self.world_size = torch.cuda.device_count()
        self.param_code_flag = specs["param_code_flag"]
        self.constraint_param_flag = specs["constraint_param_flag"]
        self.moving_avg_laplace_smooth = specs["moving_avg_library_update"]["laplace_smoothing"]
        self.moving_avg_library_update = specs["moving_avg_library_update"]["update_flag"]
        self.moving_avg_library_update_momentum = specs["moving_avg_library_update"]["update_momentum"]

        #loss here
        self.cost_matrix_reference_loss_ratio  = specs["Loss"]["cost_matrix_reference_loss_ratio"] 
        self.correspondance_reference_loss_ratio = specs["Loss"]["correspondance_reference_loss_ratio"] 
        self.type_ratio = specs["Loss"]["type_ratio"]
        self.param_ratio = specs["Loss"]["param_ratio"]
        self.ref_bia_ratio = specs["Loss"]["ref_bia_ratio"]
        self.non_type_loss_ratio = specs["Loss"]["non_type_loss_ratio"]
        self.code_regulation_weight = specs["Loss"]["code_regulation_weight"]
        self.type_attribute_list = specs["EmbeddingStructure"]["type_schema"]
        self.param_code_ratio = specs["Loss"]["param_code_ratio"]

        ### argument misnarous
        self.ref_in_argument_num = specs["EmbeddingStructure"]["ref_in_argument_num"]
        self.ref_out_argument_num = specs["EmbeddingStructure"]["ref_out_argument_num"]


        self.checkpoints = checkpoints
        self.visualize_batch = specs["visualize_batch"]
        self.device =  torch.device(rank if torch.cuda.is_available() else "cpu")
        continue_from = args["continue"]
        self.continue_from  = continue_from
        self.criteration = torch.nn.CrossEntropyLoss(reduction="none")
        self.NLLloss = torch.nn.NLLLoss(reduction="none")

        if rank == 0:
            self.writer = SummaryWriter(os.path.join(args["experiment"],"log",specs["exp_name"]))

        #we need to move model to GPU first before constructing an optimizer!
        self.optimizer = initialize_optimizer(specs["optimizer"], self.model, self.cad_embedding)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, specs["optimizer"]["learning_rate_decay"])

        self.primitive_type_mapping_table = {  #19/20/21 is belong to the start new end tokens 
            EntityType.Line: torch.tensor([0],dtype=torch.long,device=self.device),
            EntityType.Point: torch.tensor([1],dtype=torch.long,device=self.device),
            EntityType.Circle:torch.tensor([2],dtype=torch.long,device=self.device),
            EntityType.Arc: torch.tensor([3],dtype=torch.long,device=self.device),
        }

        ###additional coding for validation
        self.primitive_length_mapping = {
            0:5,
            1:3,
            2:4,
            3:6
        }
        self.param_allowance = [0,5,3,5]
        self.primitive_mapping = [[0,1,1,1,1],
             [0,1,1],
             [0,1,1,2],
             [0,1,1,2,3,3]]


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
        if continue_from is not None:
            print('continuing from "{}"'.format(continue_from))

            model_epoch = self.load_model_parameters("model_"+continue_from + ".pth")
            try:
                optimizer_epoch = self.load_optimizer("optim_"+continue_from + ".pth")
            except:
                print("Fail to load the optimizer")
            self.epoch = model_epoch + 1
            print("starting from epoch", self.epoch)
        else: 
            print("starting from scartch epoch 1")
            self.epoch = 1

        self.model.train()


    def _to(self,input_list):
        input_list = list(input_list)
        for i,_ in enumerate(input_list):
            input_list[i] = input_list[i].to(self.device,non_blocking = True)
        return input_list

    def sketch_to_tensor2(self,sketch_length, encoding_info,param_encoding):
        type_idx, type_token_batch,type_token_posi, refer_idx,refer_token_batch,refer_token_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT,batch_const_param_GT, batch_const_param_type_GT,primitive_GT, constraint_GT,primitive_length, constraint_length = encoding_info
        primitive_encoding_param, constraint_encoding_param = param_encoding

        #### transfering idx
        max_sketch_length = type_token_posi.max().item() + 1
        B = len(sketch_length)
        output_tensor = torch.zeros(B,max_sketch_length,self.model.module.detection_dim,device = self.device)
        batch_posi = torch.zeros(B,max_sketch_length,self.model.module.detection_dim,device = self.device)

        ##type
        type_embedding = self.cad_embedding.module.type_embedding(type_idx)
        output_tensor[type_token_batch,type_token_posi] += type_embedding

        ###refer
        refer_embedding = self.cad_embedding.module.reference_embedding(refer_idx)
        output_tensor[refer_token_batch,refer_token_posi] += refer_embedding

        ###primitive param
        output_tensor[primitive_param_batch,primitive_param_posi] += primitive_encoding_param * self.param_code_encoding_ratio
        if self.constraint_param_flag:
            output_tensor[constraint_param_batch,constraint_param_posi] += constraint_encoding_param * self.param_code_encoding_ratio


        ###posi
        posi_embedding = self.cad_embedding.module.posi_embedding(type_token_posi)
        batch_posi[type_token_batch,type_token_posi] += posi_embedding

        batch_posi = batch_posi.transpose(0,1) 
        output_tensor = output_tensor.transpose(0,1)
        pad_mask = torch.transpose(((output_tensor != 0).any(dim = 2)==0),dim0=0,dim1=1)
        return output_tensor,batch_posi,pad_mask
    


    def param_encoding(self,encoding_info):
        type_idx, type_token_batch,type_token_posi, refer_idx,refer_token_batch,refer_token_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT,batch_const_param_GT,batch_const_param_type_GT, primitive_GT, constraint_GT,primitive_length, constraint_length = encoding_info
        batch_prim_param_GT = batch_prim_param_GT.to(self.device,non_blocking = True)
        primitive_GT = primitive_GT.to(self.device,non_blocking = True)
        
        _,primitive_param = self.model.module.param_encode_model(batch_prim_param_GT,primitive_GT,encode_flag = True,primitive_flag = True)

        if self.constraint_param_flag:
            _,constraint_param = self.model.module.param_encode_model((batch_const_param_GT,batch_const_param_type_GT),constraint_GT,encode_flag = True,primitive_flag = False)
            return primitive_param,constraint_param

        return primitive_param,None


 
    def train_one_step(self):
        epoch_start_time = time.perf_counter()
        epoch_loss = 0
        epoch_entity_loss = 0
        epoch_reference_loss = 0
        epoch_non_type_loss = 0
        epoch_reg_loss = 0
        epoch_validity_loss = 0
        epoch_entity_type_loss = 0
        epoch_cost_matrix_reference_loss = 0
        epoch_commitment_loss = 0
        self.train_sampler.set_epoch(self.epoch)
        self.cad_embedding.module.old_counting = self.cad_embedding.module.new_counting
        for idx,input_tuple in enumerate(self.dl_train):
            #update the batch index
            self.batch = idx
            self.cad_embedding.module.batch = idx
            self.model.module.batch = idx
            #############################
            self.optimizer.zero_grad(set_to_none=True)
            sketch_GT ,sketch_length, encoding_info, GT_info = input_tuple
            encoding_info = self._to(encoding_info)
            GT_info = self._to(GT_info)

            param_encoding = self.param_encoding(encoding_info)
            input_tensors = self.sketch_to_tensor2(sketch_length, encoding_info,param_encoding)
            predicted_sketch,anxilary_loss = self.model(input_tensors,self.cad_embedding,"train")
            loss,loss_log = self.loss_calculation2(predicted_sketch[:5], GT_info,encoding_info,sketch_length,param_encoding)

            loss = loss + anxilary_loss[0] + anxilary_loss[1] + anxilary_loss[2]
            loss.backward()
            self.optimizer.step()

            #update the code 
            if self.moving_avg_library_update:
                self.library_code_update(predicted_sketch[5].detach().clone(), predicted_sketch[6].detach().clone())
            

            if self.rank == 0:
                print("epoch",self.epoch ,"batch",idx, "Loss",loss.item(), "commitment_loss",anxilary_loss[0].item(),"validity_loss",anxilary_loss[1].item(), "+ entropic loss", anxilary_loss[2].item(),"ref_reg loss", loss_log[6].item())
                print("entity loss",loss_log[4].item(),"reference loss",loss_log[1].item(),"cost_ref loss",loss_log[5].item(),"non type loss",loss_log[2].item(),"param",loss_log[0].item()- loss_log[4].item() - loss_log[5].item())
                epoch_loss += loss.detach().item()
                epoch_entity_loss += loss_log[0].detach().item()
                epoch_reference_loss += loss_log[1].detach().item()
                epoch_non_type_loss += loss_log[2].detach().item()
                epoch_reg_loss += loss_log[3].detach().item()
                epoch_entity_type_loss += loss_log[4].detach().item()
                epoch_cost_matrix_reference_loss += loss_log[5].detach().item()
                epoch_commitment_loss += anxilary_loss[0].item()
                epoch_validity_loss += anxilary_loss[1].item()

        self.optimizer.zero_grad()
        self.scheduler.step()
        epoch_end_time = time.perf_counter()
        seconds_elapsed = epoch_end_time - epoch_start_time
        print("one epoch time:",seconds_elapsed)
        if self.rank == 0:
            self.save_data() # optimizer/model etc
            self.save_log(epoch_loss,
                        epoch_entity_loss,
                        epoch_reference_loss,
                        epoch_non_type_loss,
                        epoch_reg_loss,
                        epoch_entity_type_loss,
                        epoch_validity_loss,
                        epoch_cost_matrix_reference_loss,
                        epoch_commitment_loss)
        torch.distributed.barrier()
    
    def library_code_update(self, step_code, frequency_count):   #step_code: [S2+1,E], frequency_count [S2+1] = N
        momentum = self.moving_avg_library_update_momentum
        #broadcast the input_idx and step to every other GPU
        frequency_count_list = [torch.zeros_like(frequency_count) for _ in range(self.world_size)]
        step_code_list = [torch.zeros_like(step_code) for _ in range(self.world_size)]
        torch.distributed.all_gather(step_code_list, step_code)
        torch.distributed.all_gather(frequency_count_list, frequency_count)

        #concat the input_idx and step code
        collect_frequency_count = torch.stack(frequency_count_list, dim=0).sum(dim=0)
        collect_step_code = torch.stack(step_code_list, dim=0).sum(dim=0)


        #update the mass code
        self.cad_embedding.module.mass = momentum * self.cad_embedding.module.mass  + (1 - momentum) * collect_frequency_count.unsqueeze(1) #[S2+1,1]
        self.cad_embedding.module.new_counting += collect_frequency_count.unsqueeze(1)
        self.cad_embedding.module.old_counting += collect_frequency_count.unsqueeze(1)
        #Laplace smoothing of the cluster size
        if self.moving_avg_laplace_smooth:
            n = torch.sum(self.cad_embedding.module.mass)
            self.cad_embedding.module.mass = (
                (self.cad_embedding.module.mass + 1e-5)
                / (n + self.cad_embedding.module.mass.shape[0] * 1e-5) * n)

        #update scale code
        self.cad_embedding.module.scale_code = momentum *  self.cad_embedding.module.scale_code + (1 - momentum) * collect_step_code #[S2+1,E]

        #seperate update
        self.cad_embedding.module.higher_library_embedding = self.cad_embedding.module.scale_code / self.cad_embedding.module.mass #[S2,E]



    def single_step(self):
        print("epoch {}".format(self.epoch))
        self.train_one_step()

        #evaluation
        if  self.epoch % self.validate_freq == 0 or self.epoch <= 10:
            self.validate_one_step()

        self.epoch += 1
        self.model.module.epoch = self.epoch



    def build_up_costmatrix(self,sketch,structual_logprobability,batch_idx,L0_type_index, refer_prob,GT_param,pred_param,prim_param_GT,const_param_GT,const_parm_type_GT):  #L0_one_hot: [B,S1,S2 + 1]  #to incorporate the cross entropy, the structual probability here is unnormalized and unlog!
        # decode_result [B,S1,L,E]  structual_probability [B,S1,S2+1], L0_type_index [B,S1], pred_reference_prob [B,P,S1,S1], valid_pointer_matrix [B,P,S1,S1]
        B,S0,lib_num = structual_logprobability.shape
        GT_param = GT_param.to(self.device,non_blocking=True)
        GT_component_number = len(sketch.entities) + len(sketch.constraints)
        cost_matrix = torch.zeros(GT_component_number,S0,device= self.device)   #[GT_S1,S1]
        
        #ignore the argument here 
        refer_prob_ignore_arg  = refer_prob.reshape(B,S0,self.model.module.ref_out_argument_num,S0,1).sum(4)
        entity_index_dict = {}
        #consider for the GT component number first, we need to first process the type and parameter loss, then we could proceed to the reference loss 
        type_list = []
        primitive_GT = []
        for entity_idx, entity_key in enumerate(sketch.entities):
            composite_node = sketch.entities[entity_key]
            node_type = self.primitive_type_mapping_table[composite_node.type].item()
            type_list.append(node_type)
            primitive_GT.append(node_type)
            entity_index_dict[entity_key] = entity_idx
            # type loss

            compotie_target_idx = torch.tensor(node_type,dtype=torch.long,device = self.device,requires_grad=False)
            composite_target_idx_repeat = compotie_target_idx.repeat(S0) #[S1]
            type_cost = self.NLLloss(structual_logprobability[batch_idx],composite_target_idx_repeat) #loss([S1,S2 + 1],[S1]) -> [S1] 

            # DO NOT give constant loss, but use loss_ratio to adjust the 
            cost_matrix[entity_idx] +=  self.type_ratio * type_cost 
        
        primitive_GT = torch.tensor(primitive_GT,device = self.device).long()
        constraint_GT =[]
        for constraint_idx, constraint_key in enumerate(sketch.constraints):
            constraint_idx = constraint_idx+ len(sketch.entities)
            composite_node = sketch.constraints[constraint_key]
            node_type = self.constraint_type_mapping_table[composite_node.type]
            type_list.append(node_type.item())
            constraint_GT.append(node_type.item())
            # type loss

            composite_target_idx_repeat = node_type.repeat(S0) #[S1]
            if  structual_logprobability[batch_idx].device != composite_target_idx_repeat.device:
                print("a",structual_logprobability[batch_idx].device)
                print("b",composite_target_idx_repeat.device)
            type_cost = self.NLLloss(structual_logprobability[batch_idx],composite_target_idx_repeat) #loss([S1,S2 + 1],[S1]) -> [S1] 

            # DO NOT give constant loss, but use loss_ratio to adjust the 
            cost_matrix[constraint_idx] +=  self.type_ratio * type_cost 
 
        constraint_GT = torch.tensor(constraint_GT,device = self.device).long()

        ## add parameter here
        if self.param_code_flag:
            if not self.constraint_param_flag:
                GT_param = torch.cat([GT_param,torch.zeros(len(sketch.constraints),pred_param.shape[1],device = self.device)],dim=0)

            GT_param_expand = GT_param.unsqueeze(1).repeat(1,S0,1)
            pred_param_exapnd = pred_param.unsqueeze(0).repeat(GT_component_number,1,1)
            param_cost_matrix = torch.abs(GT_param_expand - pred_param_exapnd).sum(2)

            GT_mask = torch.nonzero(torch.tensor([self.cad_embedding.module.type_to_schema[t] for t in type_list],device = self.device) == -1,as_tuple=True)
            param_cost_matrix[GT_mask] = param_cost_matrix[GT_mask] * 0
            cost_matrix += (param_cost_matrix * 0.1)
        else:
            batch_parameter = pred_param.repeat(len(sketch.entities),1)
            primitive_GT_expand = torch.repeat_interleave(primitive_GT,S0,dim=0).detach()
            primitive_GT_param_expand = torch.repeat_interleave(prim_param_GT,S0,dim=0).detach()
            batch_parameter_raw,_ = self.model.module.param_encode_model(primitive_GT_param_expand,primitive_GT_expand,param_code = batch_parameter,encode_flag = False,primitive_flag =True)
            param_idx = torch.nonzero((batch_parameter_raw == -float("Inf")).all(2).logical_not(),as_tuple = True)
            param_gt_idx = primitive_GT_param_expand[torch.nonzero((primitive_GT_param_expand != -1),as_tuple = True)]
            batch_primitive_raw_expand = batch_parameter_raw[param_idx[0],param_idx[1]]
            param_cost_expand = self.criteration(batch_primitive_raw_expand,param_gt_idx)
            param_cost_acc = torch.zeros(primitive_GT_expand.shape[0],device = self.device)
            primitive_param_cost = param_cost_acc.scatter_add_(0,param_idx[0],param_cost_expand)
            param_count = torch.zeros(primitive_GT_expand.shape[0],device = self.device)
            param_count = param_count.scatter_add_(0,param_idx[0],torch.ones_like(param_cost_expand)).reshape(-1,S0)
            primitive_param_cost = (primitive_param_cost.reshape(-1,S0) /param_count) * self.param_ratio
            cost_matrix[:len(sketch.entities)] += primitive_param_cost

            if self.constraint_param_flag:
                batch_parameter = pred_param.repeat(len(sketch.constraints),1)
                constraint_GT_expand = torch.repeat_interleave(constraint_GT,S0).detach()
                const_GT_param_expand = torch.repeat_interleave(const_param_GT,S0,dim=0).detach()
                const_GT_param_expand = torch.repeat_interleave(const_parm_type_GT,S0,dim=0).detach()
                const_GT_param_type_expand = torch.repeat_interleave(const_parm_type_GT,S0,dim=0).detach()
                batch_parameter_raw,_ = self.model.module.param_encode_model((const_GT_param_expand,const_GT_param_type_expand),None,param_code = batch_parameter,encode_flag = False,primitive_flag =False)
                param_idx = torch.nonzero((batch_parameter_raw == -float("Inf")).all(2).logical_not(),as_tuple = True)
                param_gt_idx = const_GT_param_expand[torch.nonzero((const_GT_param_type_expand[:,1] != -1),as_tuple = True)][:,1]
                batch_constraint_raw_expand = batch_parameter_raw[param_idx[0],param_idx[1]]
                param_cost_expand = self.criteration(batch_constraint_raw_expand,param_gt_idx)
                param_cost_acc = torch.zeros(constraint_GT_expand.shape[0],device = self.device)
                constraint_param_cost = param_cost_acc.scatter_add_(0,param_idx[0],param_cost_expand)
                constraint_param_cost = constraint_param_cost.reshape(-1,S0) * self.param_ratio
                cost_matrix[len(sketch.entities):] += constraint_param_cost


        for constraint_idx, constraint_key in enumerate(sketch.constraints):
            constraint_node = sketch.constraints[constraint_key]
            #here we only deal with the reference here
            constraint_idx = constraint_idx+ len(sketch.entities)
            GT_reference_index = []
            #enumerate through parameter to perform attributes loss first
            reference_count = 0
            for param in constraint_node.parameters:  #decode_result [B,S1,L,E]
                if isinstance(param, LocalReferenceParameter):
                    GT_reference_index.append(entity_index_dict[param.get_referenceMain()])
                    reference_count += 1

            #handle the reference here!
            GT_reference_tensor= torch.tensor(GT_reference_index,dtype = torch.long,device=self.device)  #[R] 
            reference_cost = torch.multiply(cost_matrix[GT_reference_tensor].unsqueeze(0),refer_prob_ignore_arg[batch_idx,:,:len(GT_reference_index)]).sum(dim=(1,2)) #mul([R,1,S1],[R,S1,S1]) ->  [R,S1,S1] ->sum [S1]
            
            #mask out the attribute cost
            cost_matrix[constraint_idx] += self.cost_matrix_reference_loss_ratio * reference_cost 
        
        return cost_matrix



    def loss_calculation2(self,predicted_sketchs, GT_info,encoding_info,sketch_length,param_encoding): 
        parameter_input_collection  , structual_probability,refer_prob, structual_onehot_collection, ref_arg_in_prob  = predicted_sketchs #[B,S1,L,E],  [B,S2 + 1,S1], #[S1,B,E], [B,S2 + 1,S1],#[B,3,S1,S1]
        structual_logprobability = torch.log(structual_probability + 1e-10)
        B,S0,Lib = structual_onehot_collection[1].shape
        type_GT,type_mask, type_batch,refer_batch_GT,refer_seq_GT,refer_dest_GT,refer_orig_GT,batch_prim_param_GT,primitive_GT = GT_info
        type_idx, type_token_batch,type_token_posi, refer_idx,refer_token_batch,refer_token_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT, batch_const_param_GT,batch_const_param_type_GT,primitive_GT, constraint_GT,primitive_length, constraint_length = encoding_info

        batch_reg_loss = torch.tensor(0.0,device = self.device)
        if self.code_regulation and len(parameter_input_collection)!=0:
            for i in range(len(parameter_input_collection)):
                batch_reg_loss  += torch.mean(torch.norm(parameter_input_collection[i], dim=2)) * self.code_regulation_weight   #[S1,B,E]



        ###build the cost matrix
        batch_cost_matrix = torch.zeros(type_GT.shape[0], S0,device=self.device)
        batch_reference_cost_matrix = torch.zeros(type_GT.shape[0], S0,device=self.device,requires_grad=False)
        batch_type_cost_matrix = torch.zeros(type_GT.shape[0], S0,device=self.device,requires_grad=False)

        ##compute for the type 
        sketch_length_torch = torch.tensor(sketch_length, device = self.device)
        type_GT_expand = torch.repeat_interleave(type_GT,S0).detach()
        batch_structual_logprobability = torch.repeat_interleave(structual_logprobability,sketch_length_torch,dim=0).flatten(0,1)
        type_cost = self.NLLloss(batch_structual_logprobability,type_GT_expand).reshape(-1,S0) # [BXGTXS0]
        batch_cost_matrix += type_cost * self.type_ratio
        batch_type_cost_matrix += type_cost * self.type_ratio

        ###compute for the parameter cost , #GT masks needed to be implmented
        if self.param_code_flag:       
            if self.constraint_param_flag:
                param_GT = torch.cat([param_encoding[0],param_encoding[1]],dim=0)
            else:
                ##break the constraint_GT  
                param_list = []
                for i in range(len(primitive_length)):
                    p_len = primitive_length[:i].sum()
                    p_limit = primitive_length[i]
                    param_list.append(param_encoding[0][p_len:p_len+p_limit])
                    param_list.append(torch.zeros(sketch_length[i] - p_limit,param_encoding[0].shape[1],device = self.device))
                param_GT = torch.cat(param_list,dim=0)

            param_GT_pad = torch.repeat_interleave(param_GT,S0,dim=0)
            batch_parameter = torch.repeat_interleave(parameter_input_collection[1],sketch_length_torch,dim=0).flatten(0,1)
            param_cost = (torch.abs(param_GT_pad - batch_parameter).sum(1).reshape(-1,S0) * self.param_code_ratio)
            batch_cost_matrix += (param_cost * type_mask.unsqueeze(1))
        else:
            primitive_length_list = primitive_length.tolist()
            primitive_mask = torch.cat([torch.tensor([1]* primitive_length_list[i] + [0] * (j - primitive_length_list[i])) for i,j in enumerate(sketch_length)],dim=0).to(self.device)
            batch_parameter = torch.repeat_interleave(parameter_input_collection[1],primitive_length,dim=0).flatten(0,1)
            primitive_GT_expand = torch.repeat_interleave(primitive_GT,S0).detach()
            primitive_GT_param_expand = torch.repeat_interleave(batch_prim_param_GT,S0,dim=0).detach()
            batch_parameter_raw,_ = self.model.module.param_encode_model(batch_prim_param_GT,primitive_GT_expand,param_code = batch_parameter,encode_flag = False,primitive_flag = True)
            param_idx = torch.nonzero((batch_parameter_raw == -float("Inf")).all(2).logical_not(),as_tuple = True)
            param_gt_idx = primitive_GT_param_expand[torch.nonzero((primitive_GT_param_expand != -1),as_tuple = True)]
            batch_primitive_raw_expand = batch_parameter_raw[param_idx[0],param_idx[1]]
            param_cost_expand = self.criteration(batch_primitive_raw_expand,param_gt_idx)
            param_cost_acc = torch.zeros(primitive_GT_expand.shape[0],device = self.device)
            param_count = torch.zeros(primitive_GT_expand.shape[0],device = self.device)
            param_cost = param_cost_acc.scatter_add_(0,param_idx[0],param_cost_expand)
            param_count = param_count.scatter_add_(0,param_idx[0],torch.ones_like(param_cost_expand)).reshape(-1,S0)
            param_cost = param_cost.reshape(-1,S0)
            primitive_mask_idx = torch.nonzero(primitive_mask,as_tuple = True)
            batch_cost_matrix[primitive_mask_idx] += (param_cost/param_count) * self.param_ratio

            if self.constraint_param_flag:
                constraint_mask = torch.cat([torch.tensor([0]* primitive_length_list[i] + [1] * (j - primitive_length_list[i])) for i,j in enumerate(sketch_length)],dim=0).to(self.device)
                batch_parameter = torch.repeat_interleave(parameter_input_collection[1],constraint_length,dim=0).flatten(0,1)
                constraint_GT_expand = torch.repeat_interleave(constraint_GT,S0).detach()
                constraint_GT_param_expand = torch.repeat_interleave(batch_const_param_GT,S0,dim=0).detach()
                const_param_type_GT_expand = torch.repeat_interleave(batch_const_param_type_GT,S0,dim=0).detach()
                batch_parameter_raw,_ = self.model.module.param_encode_model((constraint_GT_param_expand,const_param_type_GT_expand),None,param_code = batch_parameter,encode_flag = False,primitive_flag = False)
                param_idx = torch.nonzero((batch_parameter_raw == -float("Inf")).all(2).logical_not(),as_tuple = True)
                param_gt_idx = constraint_GT_param_expand[torch.nonzero((const_param_type_GT_expand[:,1] != -1),as_tuple = True)][:,1]
                batch_constraint_raw_expand = batch_parameter_raw[param_idx[0],param_idx[1]]
                param_cost_expand = self.criteration(batch_constraint_raw_expand,param_gt_idx)
                param_cost_acc = torch.zeros(constraint_GT_expand.shape[0],device = self.device)
                param_cost = param_cost_acc.scatter_add_(0,param_idx[0],param_cost_expand)
                param_cost = param_cost.reshape(-1,S0)
                constraint_mask_idx = torch.nonzero(constraint_mask,as_tuple = True)
                batch_cost_matrix[constraint_mask_idx] += param_cost * self.param_ratio

        #### reference need to considerate
        reference_cost = torch.multiply(batch_cost_matrix[refer_dest_GT].unsqueeze(1),refer_prob[refer_batch_GT,:,refer_seq_GT]).sum(2) * self.cost_matrix_reference_loss_ratio #mul([R,1,S1],[R,S1,S1]) ->  [R,S1,S1] ->sum [S1]
        refer_orig_GT_repeat = refer_orig_GT.unsqueeze(1).repeat(1,S0)
        batch_cost_matrix = batch_cost_matrix.scatter_add_(0,refer_orig_GT_repeat,reference_cost)
        batch_reference_cost_matrix = batch_reference_cost_matrix.scatter_add_(0,refer_orig_GT_repeat,reference_cost.detach())

        #### do the linear sum assignment
        cost_matrixs = torch.split(batch_cost_matrix,sketch_length,dim=0)
        type_cost_matrixs = torch.split(batch_type_cost_matrix,sketch_length,dim=0)
        reference_cost_matrixs = torch.split(batch_reference_cost_matrix,sketch_length,dim=0)
        indices = [ linear_sum_assignment(c.detach().cpu()) for c in cost_matrixs]
        indices_torch = [(torch.from_numpy(i).to(self.device,non_blocking=True), torch.from_numpy(j).to(self.device,non_blocking=True)) for i, j in indices]
        entity_loss_list = [cost_matrixs[i][idx[0],idx[1]].mean() for i,idx in enumerate(indices_torch)]
        batch_entity_loss = torch.stack(entity_loss_list,dim=0).mean()
        entity_type_loss_list = [type_cost_matrixs[i][idx[0],idx[1]].mean() for i,idx in enumerate(indices_torch)]
        batch_entity_type_loss = torch.stack(entity_type_loss_list,dim=0).mean()
        cm_ref_loss_list = [reference_cost_matrixs[i][idx[0],idx[1]].mean() for i,idx in enumerate(indices_torch)]
        batch_cost_matrix_reference_loss = torch.stack(cm_ref_loss_list,dim=0).mean()


        ###process the reference loss
        x_indices,y_indices  = zip(*indices_torch)
        #x_indices = torch.cat(x_indices,axis = 0).to(self.device)
        y_indices = torch.cat(y_indices,axis = 0)
        ref_orig_cand_idx = y_indices[refer_orig_GT]
        ref_dest_cand_idx = y_indices[refer_dest_GT]
        ref_pred_prob = refer_prob[refer_batch_GT,ref_orig_cand_idx,refer_seq_GT]
        ref_pred_logprob = torch.log(ref_pred_prob + 1e-10)
        batch_refer_count = torch.zeros(B,device = self.device)
        batch_refer_count = batch_refer_count.scatter_add_(0,refer_batch_GT,torch.ones(refer_batch_GT.shape,device = self.device))
        batch_reference_loss = torch.zeros(B,device = self.device)
        reference_loss = self.NLLloss(ref_pred_logprob, ref_dest_cand_idx) * self.correspondance_reference_loss_ratio
        batch_reference_loss = batch_reference_loss.scatter_add_(0,refer_batch_GT,reference_loss)
        batch_reference_loss =  (batch_reference_loss / batch_refer_count).mean()
        
        batch_reference_regulate_loss = torch.zeros(B,device = self.device)
        ref_pred_inscope_prob = ref_arg_in_prob[refer_batch_GT,ref_orig_cand_idx,refer_seq_GT]
        reference_regulate_loss = ref_pred_inscope_prob[...,-self.ref_out_argument_num:].sum(1) * self.ref_bia_ratio
        batch_reference_regulate_loss = batch_reference_regulate_loss.scatter_add_(0,refer_batch_GT,reference_regulate_loss)
        batch_reference_regulate_loss =  (batch_reference_regulate_loss / batch_refer_count).mean()

        ###processing the non type 
        batch_non_type_loss = torch.zeros(B,device = self.device)
        all_index = torch.ones(B,S0,device = self.device)
        all_index[type_batch,y_indices] = 0
        unslected_index = torch.nonzero(all_index,as_tuple=True) #[U] # U is the index of unselected 
        entity_target_idx = torch.tensor(self.schema_length,dtype=torch.long,device=self.device).repeat(unslected_index[0].shape[0]) #[U]
        non_type_loss = self.NLLloss(structual_logprobability[unslected_index[0],unslected_index[1]],entity_target_idx) #loss([U,S2 + 1],[U]) -> [1] Loss
        batch_non_type_loss = batch_non_type_loss.scatter_add_(0,unslected_index[0],non_type_loss)
        batch_non_type_count = torch.zeros(B,device = self.device)
        batch_non_type_count = batch_non_type_count.scatter_add_(0,unslected_index[0],torch.ones(unslected_index[0].shape,device = self.device))
        batch_non_type_loss =  (batch_non_type_loss / batch_non_type_count).mean() * self.non_type_loss_ratio

        batch_sketch_loss = batch_entity_loss + batch_reference_loss + batch_non_type_loss + batch_reg_loss + batch_reference_regulate_loss
        batch_reg_loss = batch_reg_loss/B
        return batch_sketch_loss , (batch_entity_loss,batch_reference_loss,batch_non_type_loss,  batch_reg_loss, batch_entity_type_loss,  batch_cost_matrix_reference_loss, batch_reference_regulate_loss)

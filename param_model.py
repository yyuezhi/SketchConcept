import torch
from torch._C import device
import torch.nn.functional as F
from torch import nn
from model_transformer import *
from model_embedding import *

class ParamModel(nn.Module):
    def __init__(self,specs_arch,args,specs_embed,specs,attributes_list,rank):
        super(ParamModel,self).__init__()
        self.rank = rank
        self.epoch = 0
        self.device = torch.device(rank if torch.cuda.is_available() else "cpu")
        self.encoderFClayers = specs_arch["encoderFClayers"]
        self.decoderFClayers = specs_arch["decoderFClayers"]
        self.attributeFClayers = specs_arch["attributeFClayers"]
        self.bottle_neck_dim = specs_arch["bottle_neck_dim"]
        self.angle_quantization =  specs_embed['angle_quantization']
        self.length_quantization = specs_embed["length_quantization"]
        self.coordinate_quantization =  specs_embed["coordinate_quantization"]
        self.bottle_neck_dim = specs_arch["bottle_neck_dim"]
        self.encoding_dim = specs_arch["encoding_dim"]
        self.type_attribute_list = attributes_list
        self.visualization_flag = args["visualization"]
        self.attribute_enumerate_list = ["construction_flag","angle","length","coordinate"]
        self.max_attribute_length = max([len(self.type_attribute_list[key]) for key in self.type_attribute_list]) +1
        ####initialize the FC 
        self.encoder_FC = FCBlock(self.encoderFClayers,self.bottle_neck_dim,self.max_attribute_length * self.encoding_dim,norm_flag = True)  
        self.param_code_flag = specs["param_code_flag"]
        if not specs["param_code_flag"] or self.visualization_flag:
            self.decoder_FC =  FCBlock(self.decoderFClayers,(self.max_attribute_length -1) * self.encoding_dim,self.bottle_neck_dim,norm_flag = True)    
            self.angle_FC = FCBlock(self.attributeFClayers,self.angle_quantization,self.encoding_dim)  
            self.length_FC =  FCBlock(self.attributeFClayers,self.length_quantization,self.encoding_dim)    
            self.coordinate_FC =  FCBlock(self.attributeFClayers,self.coordinate_quantization,self.encoding_dim) 
            self.construction_flag_FC =  FCBlock(self.attributeFClayers ,2 ,self.encoding_dim) 
            if specs["constraint_param_flag"]:
                self.reference_validity_FC =  FCBlock(self.attributeFClayers ,2 ,self.encoding_dim) 

        self.angle_embedding = nn.Embedding(self.angle_quantization, self.encoding_dim).to(self.device) 
        self.length_embedding = nn.Embedding(self.length_quantization, self.encoding_dim).to(self.device) 
        self.coordinate_embedding = nn.Embedding(self.coordinate_quantization, self.encoding_dim).to(self.device) 
        self.construction_flag_embedding = nn.Embedding(2, self.encoding_dim).to(self.device) 
        self.type_embedding = nn.Embedding(19, self.encoding_dim).to(self.device)  #4 primitive + 15 constraint
        if specs["constraint_param_flag"]:
            self.reference_validity_embedding = nn.Embedding(2, self.encoding_dim).to(self.device)  
        self.type_mapping_table = {  #19/20/21 is belong to the start new end tokens 
            Line: torch.tensor([0],dtype=torch.long,device=self.device),
            Point: torch.tensor([1],dtype=torch.long,device=self.device),
            Circle:torch.tensor([2],dtype=torch.long,device=self.device),
            Arc: torch.tensor([3],dtype=torch.long,device=self.device),
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
        self.primitive_param_list = ["construction_flag","coordinate","length","angle"]
        self.primitive_param_posi = torch.tensor([
           [0,1,1,1,1,-1],
           [0,1,1,-1,-1,-1],
           [0,1,1,2,-1,-1],
           [0,1,1,2,3,3],
        ],device = self.device)
        self.constraint_param_list = ["angle","length","reference_validity"]
        self.primitive_param_max_posi = torch.tensor([5,3,4,6],device = self.device)


    def type_flag_from_schema(self,name):
        bool_tensor_list = []
        for key,value in self.type_attribute_list.items():
            bool_tensor_list.append(torch.BoolTensor([ type_name == name for type_name in value]))
        bool_tensor_list.append(torch.BoolTensor([False]))  # the extra dimension for non-type 
        return torch.nn.utils.rnn.pad_sequence(bool_tensor_list).transpose(dim0 = 0,dim1 = 1).long() # [S2+1,L] return an LongTensor with 0,1 where 

    def primitive_sketch_to_tensor(self,batch_prim_param_GT,primitive_GT):
        primitive_param_table = self.primitive_param_posi[primitive_GT]
        prim_param_tensor = torch.zeros(primitive_GT.shape[0],7,self.encoding_dim,device = self.device)
        for i,attr in enumerate(self.primitive_param_list):
            param_posi = torch.nonzero(primitive_param_table == i,as_tuple = True) # output tuple pf index of attributes_mask = 1 element,extract the index which get attributes embedding
            param_idx = batch_prim_param_GT[param_posi[0],param_posi[1]] # gather the attributes #[G,E] G: the number of embedding of desired type across 
            attributes_embedding = getattr(self,attr + "_embedding")(param_idx)
            prim_param_tensor[param_posi[0],param_posi[1]] = attributes_embedding
        
        prim_param_tensor[torch.arange(primitive_GT.shape[0],device = self.device),self.primitive_param_max_posi[primitive_GT]] = self.type_embedding(primitive_GT)
        return prim_param_tensor.flatten(1,2)

    def constraint_sketch_to_tensor(self,batch_const_duo,constraint_GT):
        batch_const_param_GT,batch_const_param_type_GT = batch_const_duo
        batch_const_param_GT = batch_const_param_GT[:,1]
        batch_const_param_type_GT = batch_const_param_type_GT[:,1]

        const_param_tensor = torch.zeros(constraint_GT.shape[0],self.encoding_dim*2,device = self.device)
        for i,attr in enumerate(self.constraint_param_list):
            idx = torch.nonzero(batch_const_param_type_GT == i,as_tuple=True)
            param_idx = batch_const_param_GT[idx] # gather the attributes #[G,E] G: the number of embedding of desired type across 
            attributes_embedding = getattr(self,attr + "_embedding")(param_idx)
            const_param_tensor[:,:self.encoding_dim][idx] = attributes_embedding
        
        non_param_idx = torch.nonzero((const_param_tensor == 0).all(1),as_tuple=True)
        pparam_idx = torch.nonzero((const_param_tensor != 0).any(1),as_tuple=True)

        const_param_tensor[:,:self.encoding_dim][non_param_idx] =   self.type_embedding(constraint_GT[non_param_idx])
        const_param_tensor[:,self.encoding_dim:][ pparam_idx] =   self.type_embedding(constraint_GT[pparam_idx])
        constraint_param = torch.cat([const_param_tensor,torch.zeros(constraint_GT.shape[0],self.encoding_dim*5,device = self.device)],dim=1)
        return constraint_param


    #####here we start forward 
    def forward(self,batch_prim_param_GT,type_index_tensor,param_code = None,encode_flag = False,primitive_flag = False):
        if encode_flag:
            if primitive_flag:
                param_tensor  = self.primitive_sketch_to_tensor(batch_prim_param_GT,type_index_tensor)
            else:
                param_tensor  = self.constraint_sketch_to_tensor(batch_prim_param_GT,type_index_tensor)
            bottleneck_code = self.encoder_FC(param_tensor)
        else:
            bottleneck_code = param_code
        expand_code = self.decoder_FC(bottleneck_code)   # expand_code dim: [BXEntity,L,E]
        expand_code = expand_code.reshape(expand_code.shape[0],-1,self.encoding_dim)
        decode_output = torch.zeros_like(expand_code) #[BXEntity,L,E]
        neg_inf_value=-float('Inf')
        decode_output = torch.full(decode_output.shape,neg_inf_value,dtype = expand_code.dtype, device = self.device) 
        
        if primitive_flag:
            for attributes in self.attribute_enumerate_list:
                schema_tensor = self.type_flag_from_schema(attributes).to(self.device).float()  #[S2+1,L] L:maximal number of slot in all primitive/constrain, S2: total number of library
                attributes_mask = torch.index_select(schema_tensor,0,type_index_tensor)
                index_tuple = torch.nonzero(attributes_mask,as_tuple = True) # output tuple pf index of attributes_mask = 1 element,extract the index which get attributes embedding
                attributes_embedding = expand_code[index_tuple[0],index_tuple[1]] # gather the attributes #[G,E] G: the number of embedding of desired type across 
                attributes_result = getattr(self,attributes + "_FC")(attributes_embedding) #attributes_result #[G,O] O: the number of class the atributes have i.e. angle/length/coordinate_quantization number
                attributes_result_pad = torch.cat([attributes_result,torch.zeros(attributes_result.shape[0], self.encoding_dim - attributes_result.shape[-1],device = self.device)], dim=1) # pad with zero to make dimension equal E! cat([G,O],[G,E-O]) -> [G,E]
                decode_output[index_tuple] = attributes_result_pad  # add attributes

            return decode_output,bottleneck_code
        
        else:
            batch_const_param_GT,batch_const_param_type_GT = batch_prim_param_GT
            batch_const_param_GT = batch_const_param_GT[:,1]
            batch_const_param_type_GT = batch_const_param_type_GT[:,1]
            for i,attr in enumerate(self.constraint_param_list):
                idx = torch.nonzero(batch_const_param_type_GT == i,as_tuple=True)
                param_embedding = expand_code[idx][:,0] # gather the attributes #[G,E] G: the number of embedding of desired type across 
                attributes_result = getattr(self,attr + "_FC")(param_embedding)
                attributes_result_pad = torch.cat([attributes_result,torch.zeros(attributes_result.shape[0], self.encoding_dim - attributes_result.shape[-1],device = self.device)], dim=1) # pad with zero to make dimension equal E! cat([G,O],[G,E-O]) -> [G,E]
                idx_zero = torch.zeros_like(idx[0])
                decode_output[idx[0],idx_zero] = attributes_result_pad
            return decode_output,bottleneck_code

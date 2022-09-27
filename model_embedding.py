import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import re
from sketchgraphs.data._entity import Line,Circle,Arc, Point
from sketchgraphs.data._constraint import ConstraintType


def string_to_float(string):
    result = re.findall(r"[-+]?\d*\.\d+|\d+", string) 
    return float(result[0])


class CADEmbedding(nn.Module):
    def __init__(self, specs,rank):
        super(CADEmbedding, self).__init__()
        specs_embed = specs["EmbeddingStructure"]
        specs_arch = specs["NetworkSpecs"]

        self.global_posi_max_length = specs_embed["global_posi_max_length"]  #it is the max number of primitives+constraints that a sketch is allowed to have
        self.local_posi_max_length = specs_embed["local_posi_max_length"]   #it is the max number of attributes/embeddding that a primitives/constraints is allowed to have
        self.max_primitive_num = specs_embed["max_primitive_num"]    #the max number of primitve allowed across the dataset
        self.ref_in_argument_num = specs["EmbeddingStructure"]["ref_in_argument_num"]
        self.ref_out_argument_num = specs["EmbeddingStructure"]["ref_out_argument_num"]



        self.recon_embedding_dim = specs_arch["recon_embedding_dim"]
        self.detection_embedding_dim =  specs_arch["detection_embedding_dim"]
        self.key_embedding_dim = specs_embed["key_dim"]
        self.device = torch.device(rank if torch.cuda.is_available() else "cpu")
        self.num_library = specs["EmbeddingStructure"]["num_library"]
        self.schema_length = len(specs["EmbeddingStructure"]["type_schema"])
        self.mass_initilize = specs["moving_avg_library_update"]["mass_init"]
        self.max_detection_query = specs["EmbeddingStructure"]["max_detection_query"]
        self.angle_quantization_num =  specs["param_model"]["embeded"]['angle_quantization']
        self.length_quantization_num = specs["param_model"]["embeded"]["length_quantization"]
        self.coordinate_quantization_num =  specs["param_model"]["embeded"]["coordinate_quantization"]
        self.batch = 0


        #below embedding are hardcode embedding parameter group for encoder input! 
        #leave it there first 
        self.type_embedding = nn.Embedding(self.schema_length+3, self.detection_embedding_dim).to(self.device)  #4 primitive + 15 constraint + 3 token
        self.posi_embedding = nn.Embedding(self.global_posi_max_length * self.local_posi_max_length+1, self.detection_embedding_dim).to(self.device)   #max 10 attributes
        self.reference_embedding = nn.Embedding(51 , self.detection_embedding_dim).to(self.device)  
        

        if specs["moving_avg_library_update"]["update_flag"]:
            self.register_buffer("mass",torch.full((self.num_library,1),self.mass_initilize, device = self.device))
            self.register_buffer("old_counting",torch.full((self.num_library,1),0.0, device = self.device))
            self.register_buffer("new_counting",torch.full((self.num_library,1),0.0, device = self.device))
            self.register_buffer("scale_code",torch.zeros(self.num_library,self.key_embedding_dim[1]))
            nn.init.normal_(self.scale_code)
            print("Initialize with moving avg with library code")  

        #input for detection decoder
        self.detection_query_embed = nn.Embedding(self.max_detection_query+1,self.detection_embedding_dim).to(self.device) 



        setattr(self, "Lower_library_embedding",nn.Embedding(self.schema_length,self.key_embedding_dim[0]).to(self.device))
        setattr(self, "Lower_library_nontype",nn.Embedding(1,self.key_embedding_dim[0]).to(self.device)) 

        self.type_attributes_schema =  specs_embed["type_schema"]
        self.type_to_string_table = {
            Line: "Line",
            Point: "Point",
            Circle: "Circle",
            Arc: "Arc",
            ConstraintType.Coincident: "Coincident",
            ConstraintType.Distance: "Distance",
            ConstraintType.Horizontal : "Horizontal",  
            ConstraintType.Parallel : "Parallel",
            ConstraintType.Vertical : "Vertical",
            ConstraintType.Tangent : "Tangent",
            ConstraintType.Length : "Length",
            ConstraintType.Perpendicular : "Perpendicular",
            ConstraintType.Midpoint : "Midpoint",
            ConstraintType.Equal : "Equal",
            ConstraintType.Diameter : "Diameter",
            ConstraintType.Radius : "Radius",
            ConstraintType.Angle : "Angle",
            ConstraintType.Concentric : "Concentric",
            ConstraintType.Normal : "Normal",
        }
        self.type_to_schema = {  # 19/20/21 is belong to the start new end tokens
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: -1,
            5: 4,
            6: -1,
            7: -1,
            8: -1,
            9: -1,
            10: 4,
            11: -1,
            12: -1,
            13: -1,
            14: 4,
            15: 4,
            16: 5,
            17: -1,
            18: -1,
        }
        self.initialize_embedding_weight()

        ##here we initialize the weight for mass and scale weight 
        if  specs["moving_avg_library_update"]["update_flag"]:
            #unit vector initialize as 1
            self.register_buffer("higher_library_embedding",self.scale_code.clone())
        
    def initialize_embedding_weight(self):
        for p in self.parameters():
            nn.init.normal_(p)

 
    def type_flag_from_schema(self,name):
        bool_tensor_list = []
        for key,value in self.type_attributes_schema.items():
            bool_tensor_list.append(torch.BoolTensor([ type_name == name for type_name in value]))
        bool_tensor_list.append(torch.BoolTensor([False]))  # the extra dimension for non-type 
        return torch.nn.utils.rnn.pad_sequence(bool_tensor_list).transpose(dim0 = 0,dim1 = 1).long() # [S2+1,L] return an LongTensor with 0,1 where 

    def coordinate_class_to_value(self,classification):
        return (classification + 0.5)/self.coordinate_quantization_num * 2.00001 - 1

    def length_class_to_value(self,classification):
        return (classification+ 0.5)/self.length_quantization_num * 2.00001
    
    def angle_class_to_value(self,classification):
        return (classification + 0.5)/self.angle_quantization_num * (2*math.pi+1e-6)

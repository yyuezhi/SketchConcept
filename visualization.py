import numpy as np
import torch
from sketchgraphs.data._entity import Line, Point, Circle, Arc
from sketchgraphs.data._constraint import Constraint,  LocalReferenceParameter, QuantityParameter
from sketchgraphs.data.sketch import Sketch
import collections
from scipy.optimize import linear_sum_assignment
from model_transformer import *
SKETCH_VISUALIZATION_DIR = "sketch_visualization"
SKETCH_REWRITE_DIR = "sketch_rewrite"
MODEL_PARAM_DIR = "network_param"
import os
from sketchgraphs.data import flat_array
import pickle
import networkx as nx

def initialize_network(self,type,reference,predicted_idx_list,sketch):
    nontype = self.schema_length #currently 19
    type_list = type.cpu().tolist()
    predicted_idx_list = predicted_idx_list.cpu().tolist()
    refer_dest = reference.argmax(2).cpu()

    G = nx.MultiDiGraph()
    node_counter = 0
    #initialize node
    G.add_node(-1,type = -1,predicted_idx = -1)  ##add a dummy node, to account for constraint pointing to non type 
    for i,node_type in enumerate(type_list):
        if node_type != nontype:
            G.add_node(i, type = node_type,predicted_idx = i)   ###predicted_idx here is the index of predicted element
            node_counter  += 1
    

    #initialize edge
    for i,node_type in enumerate(type_list):
        if node_type != nontype:
            name = list(self.type_attribute_list.keys())[G.nodes[i]["type"]]
            
            #if can not match, then use default
            try:
                if i in  predicted_idx_list and predicted_idx_list.index(i) >= len(sketch.entities): 
                    match_idx = predicted_idx_list.index(i) - len(sketch.entities) + 1
                    if self.constraint_type_mapping_table[sketch.constraints["c_"+str(match_idx)].type].item() != G.nodes[i]["type"]:
                        num_constraint = self.type_attribute_list[name].count("pointer")+self.type_attribute_list[name].count("pointer_valid")
                    else:
                        num_constraint = len([param for param in sketch.constraints["c_"+str(match_idx)].parameters if isinstance(param, LocalReferenceParameter)])
                else:
                    num_constraint = self.type_attribute_list[name].count("pointer")+self.type_attribute_list[name].count("pointer_valid")
            except:
                num_constraint = self.type_attribute_list[name].count("pointer")+self.type_attribute_list[name].count("pointer_valid")
            for k,d in enumerate(refer_dest[i].tolist()[:num_constraint]):
                destin = int(d)
                if destin in G.nodes:
                    G.add_edge(i,destin,key_in = d,edge_idx = k) ##key_in here is just to avoid duplicated edge, edge_idx is here to denote the number of pointer
                else:
                    G.add_edge(i,-1,key_in = 0,edge_idx = k)
    
    ###reindex the node from index of matched predicted element to continuous num from 0 -> x
    G = index_update(G)
    return G





def index_update(G):
    nodes = list(G.nodes)[1:]
    mapping = dict(zip(nodes,list(range(len(nodes)))))
    mapping[-1] = -1
    H = nx.relabel_nodes(G,mapping,copy=True)
    return H


def constraint_node2graph(self,graph,ref_arg_in_argmax,ref_arg_out_argmax,batch_idx):
    old_nodes = copy.deepcopy(dict(graph.nodes(data=True)))
    max_primitive_type_idx = len(self.primitive_type_mapping_table)-1
    Q = self.max_detection_query
    total_in_arg_num =  Q * self.ref_in_argument_num
    total_out_arg_num =  Q * self.ref_out_argument_num
    out_arg_base = 100 #hardcode large number
    in_arg_base = 200
    A = self.max_abstruction

    edge_key = []
    for key in dict(graph.nodes(data=True)).keys():
        node = old_nodes[key]
        if node["type"] <= max_primitive_type_idx:
            continue
        else:
            edge_key.append(key)
            del old_nodes[key]  ##delete the old node so that there is only new node left
    

    #edge_key = edge_key[1:]   ##Why do we need this?
    new_graph = nx.MultiDiGraph()

    ### We first add primitive in to it
    for node in old_nodes:
        value = old_nodes[node]
        new_graph.add_node(node,attr = "primitive",type = value["type"],predicted_idx = value["predicted_idx"])
    
    ## We then add the in and out argument nodes for each query
    for i in range(total_out_arg_num):
        new_graph.add_node(-out_arg_base-i,attr = "out_arg",type = -2,predicted_idx = -2)

    for i in range(total_in_arg_num):
        new_graph.add_node(-in_arg_base-i,attr = "in_arg",type = -2,predicted_idx = -2)
    prim_nodes = list(old_nodes.keys())

    repeated_cross_edge = []
    for j,key in enumerate(edge_key):
        edge_dest = []
        node = graph.nodes[key]   # for each edge node
        for edge in list(graph.out_edges(key)):
            edge_dest.append(edge[1])
        if len(edge_dest) == 1:  ## add one dummy for easier processing
            edge_dest.append(edge_dest[0])

        ###If this edge is not link to the primitive nodes, then discard it!
        discard = False
        for d in edge_dest:
            if d not in prim_nodes:
                discard = True
        if discard:
            continue

        d1 = int(new_graph.nodes[edge_dest[0]]["predicted_idx"] / A) ## the query where the constraint  element is located
        dc = int(node["predicted_idx"]/A)           ## the query where the first pointee element is located
        d2 = int(new_graph.nodes[edge_dest[1]]["predicted_idx"] / A) # the query where the second pointee element is located
        
        constraint_edge = []  
        if d1 == d2 and d2 == dc:##in the same concept
            new_graph.add_edge(*edge_dest,predicted_idx = node["predicted_idx"],type = node["type"])
            new_graph.add_edge(*list(reversed(edge_dest)),predicted_idx = node["predicted_idx"],type = node["type"])
            continue
        if d1 == dc:
            constraint_edge.append(edge_dest[0])
        else:
            ###
            arg1out = ref_arg_in_argmax[node["predicted_idx"],0].item()  ##[-1,-2]
            arg1out_node = -out_arg_base - (self.ref_out_argument_num* dc)+arg1out +1
            constraint_edge.append(arg1out_node)
            ###first insert argument
            arg1in = ref_arg_out_argmax[(self.ref_in_argument_num* dc)-arg1out -1].item()
            arg1in_node = -in_arg_base -arg1in
            if set((arg1out_node,arg1in_node)) not in repeated_cross_edge:
                repeated_cross_edge.append(set((arg1out_node,arg1in_node)))
                new_graph.add_edge(*[arg1out_node,arg1in_node],predicted_idx = -5,type = node["type"])
            if set((arg1in_node,edge_dest[0])) not in repeated_cross_edge:
                repeated_cross_edge.append(set((arg1in_node,edge_dest[0])))
                new_graph.add_edge(*[arg1in_node,edge_dest[0]],predicted_idx = new_graph.nodes[edge_dest[0]]["predicted_idx"],type = node["type"])


        if d2 == dc:
            constraint_edge.append(edge_dest[1])
        else:
            ###
            arg1out = ref_arg_in_argmax[node["predicted_idx"],1].item()  ##[-1,-2]
            arg1out_node = -out_arg_base - (self.ref_out_argument_num* dc)+arg1out +1
            constraint_edge.append(arg1out_node)
            ###first insert argument

            arg1in = ref_arg_out_argmax[(self.ref_in_argument_num* dc)-arg1out -1].item()
            arg1in_node = -in_arg_base -arg1in
            if set((arg1out_node,arg1in_node)) not in repeated_cross_edge:
                repeated_cross_edge.append(set((arg1out_node,arg1in_node)))
                new_graph.add_edge(*[arg1out_node,arg1in_node],predicted_idx = -5,type = node["type"])
            if set((arg1in_node,edge_dest[1])) not in repeated_cross_edge:
                repeated_cross_edge.append(set((arg1in_node,edge_dest[1])))
                new_graph.add_edge(*[arg1in_node,edge_dest[1]],predicted_idx = new_graph.nodes[edge_dest[1]]["predicted_idx"],type = node["type"])
        new_graph.add_edge(*constraint_edge,predicted_idx = node["predicted_idx"],type = node["type"])
        new_graph.add_edge(*list(reversed(constraint_edge)),predicted_idx = node["predicted_idx"],type = node["type"])
    isolate_node = [i for i in nx.isolates(new_graph) if i <0 ]
    new_graph.remove_nodes_from(isolate_node)
    return new_graph

def prediction_to_sketch(self,predicted_sketchs, sketches,input_param,encoding_info):
    type_idx, type_token_batch,type_token_posi, refer_idx,refer_token_batch,refer_token_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT, batch_const_param_GT,batch_const_param_type_GT,primitive_GT, constraint_GT,primitive_length, constraint_length = encoding_info
    parameter_collection,    structual_probability, refer_prob,structual_onehot_collection,cross_ref_detail = predicted_sketchs[0], predicted_sketchs[1].clone().detach() ,predicted_sketchs[2].clone().detach(),predicted_sketchs[3],predicted_sketchs[4] #[B,S1,L,E],  _ ,[B,S1,S2+1],  [B,P,S1,S1], [B,P,S1,1] P = 3
    ref_arg_in_prob,ref_arg_out_prob,ref_arg_in_crosscope_prob_reverse = cross_ref_detail

    ###Initialize the basic parameters
    L = self.max_abstruction
    B = ref_arg_in_prob.shape[0]
    Q = self.max_detection_query
    C = Q * L
    arg_in_num = self.ref_in_argument_num
    arg_out_num = self.ref_out_argument_num

    ref_inscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,L,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_crossscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,arg_in_num,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_dia_idx_tensor = torch.ones_like(ref_arg_in_prob)
    ref_dia_idx_tensor[:,ref_inscope_dia_idx_not[0],:,ref_inscope_dia_idx_not[1]] = 0
    ref_dia_idx_tensor[:,ref_crossscope_dia_idx_not[0],:,C + ref_crossscope_dia_idx_not[1]] = 0
    ref_idx_not = torch.nonzero(ref_dia_idx_tensor,as_tuple=True)
    ref_arg_in_argmax = ref_arg_in_prob[ref_idx_not].reshape(B,Q,L,arg_out_num,L+arg_in_num).argmax(4) #[B,Q,L,matptr,L*A+arg]
    ref_arg_in_argmax = torch.where(ref_arg_in_argmax<L,ref_arg_in_argmax,-ref_arg_in_argmax + L-1).flatten(1,2)
    ref_arg_out_argmax = ref_arg_out_prob.argmax(2)

    structual_logprobability = torch.log(structual_probability + 1e-10)
    parameter_L0_output = parameter_collection[1]
    B,S0,E = parameter_L0_output.shape  #S1
    L0_struc_prob = structual_probability #[B,S1,S2 + 1]
    type_index = L0_struc_prob.argmax(dim=2) #[B,S1]
    lib_argmax = structual_onehot_collection[0].argmax(2)
    ###temporary modification
    param_decode,_ = self.model.module.param_encode_model(batch_prim_param_GT,type_index.flatten(),param_code = parameter_collection[1].flatten(0,1),encode_flag = False,primitive_flag =True)



    param_decode = param_decode.reshape(B,-1,param_decode.shape[1],param_decode.shape[2])

    param_GT_list = []
    param_list = []
    param_const_list = []
    param_const_type_list = []
    for i in range(len(primitive_length)):
        p_len = primitive_length[:i].sum()
        p_limit = primitive_length[i]
        c_len = constraint_length[:i].sum()
        c_limit = constraint_length[i]
        if self.constraint_param_flag:
            param_list.append(torch.cat([input_param[0][p_len:p_len+p_limit],input_param[1][c_len:c_len+c_limit]],dim=0))
        else:
            param_list.append(torch.cat([input_param[0][p_len:p_len+p_limit],torch.zeros(c_limit,input_param[0].shape[1],device=self.device)],dim=0))  
        param_GT_list.append(batch_prim_param_GT[p_len:p_len+p_limit])
        param_const_list.append(batch_const_param_GT[c_len:c_len+c_limit])
        param_const_type_list.append(batch_const_param_type_GT[c_len:c_len+c_limit])

    batch_sketch_list = []
    batch_belong_list = []
    batch_type_list = []
    batch_corr_list = []
    batch_graph_list = []
    for batch_idx in range(len(sketches)): #B
        sketch = sketches[batch_idx]
        cost_matrix = self.build_up_costmatrix(sketches[batch_idx],structual_logprobability,batch_idx, type_index, refer_prob,param_list[batch_idx],parameter_L0_output[batch_idx],param_GT_list[batch_idx],param_const_list[batch_idx],param_const_type_list[batch_idx])#[GT_S1,S1]

        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_idx_np, col_idx_np = linear_sum_assignment(cost_matrix_np)
        row_idx = torch.tensor(row_idx_np,dtype = torch.long, device=self.device)
        col_idx = torch.tensor(col_idx_np,dtype = torch.long, device=self.device)

        ### we shall first deal with the parameter then we deal with the structure and reference, temporaiy implementation for L1 only        
        ###Then we aim to build the structure and reference
        graph = self.initialize_network(type_index[batch_idx],refer_prob[batch_idx],col_idx,sketch)

        ###during the next stage, we convert the Graph in to sketch
        sketch,belong,ttype,corr = self.graph_to_sketch(graph,param_decode[batch_idx],lib_argmax[batch_idx],col_idx_np)
        
        ### Then, we convert the initialized graph's edge representation from node to edge 
        g = self.constraint_node2graph(graph,ref_arg_in_argmax[batch_idx],ref_arg_out_argmax[batch_idx],batch_idx)
        batch_sketch_list.append(sketch)
        batch_belong_list.append(belong)
        batch_type_list.append(ttype)
        batch_corr_list.append(corr)
        batch_graph_list.append(g)
    return batch_sketch_list,batch_belong_list,batch_type_list,batch_corr_list,batch_graph_list

        








def library_processing(self, predicted_sketch,library_pointer,library_argument):
    parameter_input_collection, structual_L0_probability,refer_prob,structual_onehot_collection,cross_ref_detail = predicted_sketch
    L1_argmax = structual_onehot_collection[0].argmax(2)
    ref_arg_in_prob,ref_arg_out_prob,ref_arg_in_crosscope_prob_reverse = cross_ref_detail
    max_ptr_num = self.ref_out_argument_num      #the max number of the previous library
    new_arg_in_num = self.model.module.ref_in_argument_num   #the library building level arugment number
    Q = self.max_detection_query  #number of query
    L = self.max_abstruction # number of abstraction
    C = Q*L #candidate number
    B = L1_argmax.shape[0]
    new_library_num = self.num_library


    ref_inscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,L,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_crossscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,new_arg_in_num,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_dia_idx_tensor = torch.ones_like(ref_arg_in_prob)
    ref_dia_idx_tensor[:,ref_inscope_dia_idx_not[0],:,ref_inscope_dia_idx_not[1]] = 0
    ref_dia_idx_tensor[:,ref_crossscope_dia_idx_not[0],:,C + ref_crossscope_dia_idx_not[1]] = 0
    ref_idx_not = torch.nonzero(ref_dia_idx_tensor,as_tuple=True)
    ref_arg_in_argmax = ref_arg_in_prob[ref_idx_not].reshape(B,Q,L,max_ptr_num,L+new_arg_in_num).argmax(4) #[B,Q,L,matptr,L*A+arg]
    ref_arg_in_argmax = torch.where(ref_arg_in_argmax<L,ref_arg_in_argmax,-ref_arg_in_argmax + L-1)
    ref_arg_in_cross_rev_argmax = torch.remainder(ref_arg_in_crosscope_prob_reverse.argmax(1),L).reshape(B,Q,new_arg_in_num) #[B,LXA,QXarg] -> [B,Q,arg]
    for i in range(new_library_num):
        library_index = torch.nonzero(L1_argmax == i,as_tuple=True)
        if library_index[0].shape[0] !=0:
            library_pointer[i] = ref_arg_in_argmax[library_index[0][0],library_index[1][0]]
            library_argument[i] = ref_arg_in_cross_rev_argmax[library_index[0][0],library_index[1][0]]
        else:
            library_pointer[i] = torch.ones(L,max_ptr_num,device = self.device) * 100
            library_argument[i] = torch.ones(self.model.module.ref_out_argument_num,device = self.device) * 100 

    return library_pointer,library_argument

def save_visualize_sketch(self,batch_sketch,batch_belong,batch_type,batch_corr,library_pointer,library_argument,library_structure,batch_graph_json,library_select_list,filename):
    offsets_sketch, sequence_data_sketch = flat_array.raw_list_flat(batch_sketch)
    flat_data_sketch = flat_array.pack_list_flat(offsets_sketch,sequence_data_sketch)
    sketch_length = np.array([len(sketch.entities) + len(sketch.constraints) for sketch in batch_sketch])
    flat_dict_sketch = flat_array.pack_dictionary_flat({'sequences': flat_data_sketch,'sequence_lengths' : sketch_length})
    np.save(os.path.join(self.output_dir,SKETCH_VISUALIZATION_DIR, filename), flat_dict_sketch, allow_pickle=False)
    np.savez(os.path.join(self.output_dir,SKETCH_VISUALIZATION_DIR, filename+ "_stat"),pointer = library_pointer, arguement = library_argument,structure = library_structure,\
             belong = batch_belong,type = batch_type,corr = batch_corr,graph = batch_graph_json,lib_select = library_select_list, allow_pickle = True)
    print("saving sketch visualize done!")

@torch.no_grad()
def sketch_visualization(self):
    library_pointer = torch.zeros(self.num_library,self.max_abstruction,self.model.module.ref_out_argument_num) 
    library_argument = torch.zeros(self.num_library,self.model.module.ref_in_argument_num)
    library_code = getattr(self.cad_embedding.module,"higher_library_embedding").to(self.device).unsqueeze(0)
    predicted_lib_type = self.model(library_code,self.cad_embedding,"library_expo")
    predicted_type = predicted_lib_type.argmax(dim=2).reshape(self.num_library,-1)
    batch_sketch_list = []
    batch_belong = []
    batch_type = []
    batch_corr = []
    batch_graph = []
    library_select_list = []
    print("visualization process start for epoch", self.epoch)
    for idx,input_tuple in enumerate(self.dl_validation):
        sketch_GT,sketch_length, encoding_info, GT_info = input_tuple
        encoding_info = self._to(encoding_info)
        GT_info = self._to(GT_info)
 
        param_encoding= self.param_encoding(encoding_info)
        input_tensors = self.sketch_to_tensor2(sketch_length, encoding_info,param_encoding)
        predicted_sketch = self.model(input_tensors,self.cad_embedding,"visualize")
        library_select_list.append(predicted_sketch[3][0].argmax(2))
        library_pointer,library_argument = self.library_processing(predicted_sketch,library_pointer,library_argument)
        sketch_list,belong,ttype,corr_predicted_idx,graph = self.prediction_to_sketch(predicted_sketch,sketch_GT,param_encoding,encoding_info)
        batch_sketch_list.extend(sketch_list)
        batch_graph.extend(graph)
        batch_belong.extend(belong)
        batch_type.extend(ttype)
        batch_corr.extend(corr_predicted_idx)
        print("visualize_batch",idx)
        if self.visualize_batch == idx:
            break

    library_select_list = torch.cat(library_select_list,dim=0).cpu().numpy()
    batch_belong = np.array(batch_belong,dtype= np.object)
    batch_type = np.array(batch_type,dtype= np.object)
    batch_corr = np.array(batch_corr,dtype= np.object)
    library_pointer = library_pointer.cpu().numpy()
    library_argument= library_argument.cpu().numpy()
    predicted_type = predicted_type.cpu().numpy()
    if self.rank == 0:
        self.save_visualize_sketch( batch_sketch_list, batch_belong,batch_type,batch_corr,library_pointer,library_argument,predicted_type,batch_graph,library_select_list,self.expname+ "_epoch_"+str(self.epoch)+"_sketch")
    print("sketch rewrite process done!")

    

def graph_to_sketch(self,graph,param_code,L1_argmax,predicted_idx_match_list):
    entities = collections.OrderedDict()
    constraints = collections.OrderedDict()
    L = self.max_abstruction
    predicted_idx_match_list = predicted_idx_match_list.tolist()
    L1_argmax = L1_argmax.cpu().tolist()
    max_candidate_num = L* self.max_detection_query
    node_idx = -1
    constraint_idx = 1
    predicted_idx_list = []
    correspond_matched_idx = []
    node_map = {}
    group_type = np.zeros((len(graph.nodes)-1,2)) ##the first dim of 2 dim is the node type, the second is the query belong to 
    #first process the primitive first
    for node in graph.nodes: #S1
        if node == -1:  #ignore the dummy mode
            continue
        candidate_type = graph.nodes[node]["type"]
        predicted_idx = graph.nodes[node]["predicted_idx"]  #the predicted element's index
        if candidate_type < 4: #this is primitive type 
            try:
               idx = predicted_idx_match_list.index(predicted_idx) ### decide whether this predicted index is matched with GT
            except ValueError:
               idx = -1
            correspond_matched_idx.append(idx)
            predicted_idx_list.append(predicted_idx)  # group_belong to this matched node
            #temporary things
            node_idx += 1
            group_type[node_idx][0] = candidate_type
            group_type[node_idx][1] = L1_argmax[int(predicted_idx/L)]

            type, schema = list(self.cad_embedding.module.type_attributes_schema.items())[candidate_type]
            entityId = str(node_idx)
            node_map[node] = entityId
            # simplify version of the group belong

            #done
            if type == "Line":
                obj = Line(entityId)
                #construction flag
                obj.isConstruction = bool(param_code[predicted_idx,0,0:2].argmax().cpu())

                #coordinate
                predicted_value = self.cad_embedding.module.coordinate_class_to_value(param_code[predicted_idx,1:5,0:self.cad_embedding.module.coordinate_quantization_num].argmax(dim=1).cpu().numpy()) #[4]
                obj.pntX, obj.pntY = predicted_value[:2]
                length = np.linalg.norm(predicted_value[2:] - predicted_value[:2])
                obj.dirX, obj.dirY = (predicted_value[2:] - predicted_value[:2]) / length #B-A ..
                obj.startParam = 0 
                obj.endParam = float(length)

                entities[entityId] = obj

            elif type == "Circle":
                obj = Circle(entityId)
                #construction flag
                obj.isConstruction = bool(param_code[predicted_idx,0,0:2].argmax().cpu())
                #coordinate value
                obj.xCenter, obj.yCenter = self.cad_embedding.module.coordinate_class_to_value(param_code[predicted_idx,1:3,0:self.cad_embedding.module.coordinate_quantization_num].argmax(dim=1).cpu().numpy()) #[2]

                #radius
                obj.radius = float(self.cad_embedding.module.length_class_to_value(param_code[predicted_idx,3,0:self.cad_embedding.module.length_quantization_num].argmax().cpu().numpy())) 

                entities[entityId] = obj
            
            elif type == "Point":
                obj = Point(entityId)
                #construction flag
                obj.isConstruction = bool(param_code[predicted_idx,0,0:2].argmax().cpu().numpy())
                #coordinate value
                obj.x, obj.y = self.cad_embedding.module.coordinate_class_to_value(param_code[predicted_idx,1:3,0:self.cad_embedding.module.coordinate_quantization_num].argmax(dim=1).cpu().numpy())

                entities[entityId] = obj
            
            elif type == "Arc":
                obj = Arc(entityId)
                #construction flag
                obj.isConstruction = bool(param_code[predicted_idx,0,0:2].argmax().cpu().numpy())

                #coordinate first
                obj.xCenter, obj.yCenter = self.cad_embedding.module.coordinate_class_to_value(param_code[predicted_idx,1:3,0:self.cad_embedding.module.coordinate_quantization_num].argmax(dim=1).cpu().numpy())

                #the radius
                obj.radius = float(self.cad_embedding.module.length_class_to_value(param_code[predicted_idx,3,0:self.cad_embedding.module.length_quantization_num].argmax().cpu().numpy()))

                #the angle
                obj.startParam, obj.endParam = self.cad_embedding.module.angle_class_to_value(param_code[predicted_idx,4:6,0:self.cad_embedding.module.angle_quantization_num].argmax(dim=1).cpu().numpy())

                entities[entityId] = obj




    for node in graph.nodes:
        if node == -1:
            continue
        candidate_type = graph.nodes[node]["type"]
        predicted_idx = graph.nodes[node]["predicted_idx"]
        if candidate_type >= 4:
            try:
               idx = predicted_idx_match_list.index(predicted_idx)
            except ValueError:
               idx = -1
            correspond_matched_idx.append(idx)
            predicted_idx_list.append(predicted_idx)
            node_idx += 1
            group_type[node_idx][0] = candidate_type

            group_type[node_idx][1] = L1_argmax[int(predicted_idx/L)]

            type, schema = list(self.cad_embedding.module.type_attributes_schema.items())[candidate_type]


            constraint_type = list(self.cad_embedding.module.type_to_string_table.keys())[list(self.cad_embedding.module.type_to_string_table.values()).index(type)]
            constraint_ent_id = 'c_%i' % constraint_idx
            constraint_idx += 1
            params = []

            reference_count = -1
            for schema_idx, name in enumerate(schema):
                if name in ["pointer", "pointer_valid"] and len(list(graph.out_edges(node))) >= reference_count+2:  # here we only consider refer to an entity as a whole, not by part
                    reference_count = reference_count + 1
                    try:
                        reference_entity = node_map[list(graph.out_edges(node))[reference_count][1]]
                    except KeyError as e:
                        continue
                    params.append(LocalReferenceParameter("local" + str(reference_count), reference_entity))

            constraints[constraint_ent_id] = Constraint(constraint_ent_id, constraint_type, params)
    predicted_idx_list = np.asarray(predicted_idx_list)
    correspond_matched_idx = np.asarray(correspond_matched_idx)
    return Sketch(entities=entities, constraints=constraints),predicted_idx_list,group_type,correspond_matched_idx


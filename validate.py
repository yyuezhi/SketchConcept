import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import os
from sketchgraphs.data._constraint import  LocalReferenceParameter
SKETCH_VISUALIZATION_DIR = "sketch_visualization"
@torch.no_grad()
def validate_one_step(self):
    #evaluation
    type_total_num = 0
    type_right_num = 0
    refer_total_num = 0
    refer_right_num = 0
    refer_inscope_num = 0
    refer_inscope_right_num = 0
    non_type_total_num = 0
    non_type_right_num = 0


    self.model.eval()
    self.validation_sampler.set_epoch(0)

    new_lib_num = self.num_library
    upper_decompose_type = {}
    ###
    pred_prim_count = 0
    pred_const_count = 0
    GT_prim_count = 0
    GT_const_count = 0
    prim_match_count = 0
    const_match_count = 0
    print("validation process start for epoch", self.epoch)
    for idx,input_tuple in enumerate(self.dl_validation):
        sketch_GT,sketch_length, encoding_info, GT_info = input_tuple
        encoding_info = self._to(encoding_info)
        GT_info = self._to(GT_info)
 
        param_encoding = self.param_encoding(encoding_info)
        input_tensors = self.sketch_to_tensor2(sketch_length, encoding_info,param_encoding)
        predicted_sketch = self.model(input_tensors,self.cad_embedding,"validate")

        type_stat, refer_stat, non_type_stat,matching_result = self.accuracy_evaluation(predicted_sketch,sketch_GT,param_encoding,encoding_info)

        
        type_total_num += type_stat[0]
        type_right_num += type_stat[1]
        refer_total_num += refer_stat[0]
        refer_right_num += refer_stat[1]
        refer_inscope_num += refer_stat[2]
        refer_inscope_right_num += refer_stat[3] 
        non_type_total_num += non_type_stat[0]
        non_type_right_num += non_type_stat[1]

        pred_prim_count += matching_result[0]
        pred_const_count += matching_result[1]
        GT_prim_count += matching_result[2]
        GT_const_count+= matching_result[3]
        prim_match_count+= matching_result[4]
        const_match_count += matching_result[5]

        if idx == 10:
            break
        if self.rank == 0:
            print("validation", self.epoch, "batch", idx)





    library_code = getattr(self.cad_embedding.module,"higher_library_embedding").to(self.device)
    batch_nontype_library_code = library_code.unsqueeze(0) #[B,S1,E]
    predicted_sketch_non_type = self.model(batch_nontype_library_code,self.cad_embedding,"library_expo")
    predicted_type = predicted_sketch_non_type.argmax(dim=2)[0].reshape(library_code.shape[0],-1) 
    for i in range(0,new_lib_num):
        upper_decompose_type[i] = predicted_type[i]

    if self.rank == 0:
        epoch_type_accuracy = type_right_num / float(type_total_num)
        epoch_refer_accuracy = refer_right_num / float(refer_total_num)
        epoch_refer_right_inscope_accuracy = refer_inscope_right_num / float(refer_right_num)
        epoch_non_type_accuracy = non_type_right_num / float(non_type_total_num)
        # prim_precision = prim_match_count/float(GT_prim_count)
        # prim_recall = prim_match_count/float(pred_prim_count)
        # const_precision = const_match_count/float(GT_const_count)
        # const_recall = const_match_count/float(pred_const_count)
        # prim_F1 = 2*prim_precision*prim_recall/(prim_precision+prim_recall)
        # const_F1 = 2*const_precision*const_recall/(const_precision+const_recall)

        # print("validation process end")
        # print("Primitive F1",prim_F1)
        # print("Constraint F1",const_F1)
        print("________________Basic statistics_________________")
        print("epoch",self.epoch,"type_accuracy",epoch_type_accuracy)
        print("epoch",self.epoch,"refer_accuracy",epoch_refer_accuracy)
        print("epoch",self.epoch,"refer_right_inscope_accuracy",epoch_refer_right_inscope_accuracy)
        print("epoch",self.epoch,"non_type_accuracy",epoch_non_type_accuracy)
        self.save_epoch_accuracy_log(epoch_type_accuracy,
                                epoch_refer_accuracy,
                                epoch_non_type_accuracy)
    torch.distributed.barrier()
    self.model.train()

def compare_primitive(self,gt_tuple,pred_tuple):
    if gt_tuple[0] != pred_tuple[0]:
        return False
    schema = self.primitive_mapping[gt_tuple[0]]
    def param_check():
        for i in range(1,len(schema)+1):  
            if schema[i-1] == 3:
                if abs(gt_tuple[i] - pred_tuple[i]) > self.param_allowance[schema[i-1]] and (abs(gt_tuple[i] - pred_tuple[i]) - 30) > self.param_allowance[schema[i-1]]:
                    return False            
            else:
                if abs(gt_tuple[i] - pred_tuple[i]) > self.param_allowance[schema[i-1]]:
                    return False
        return True

    ## keep in mind of the duality exist in line
    if gt_tuple[0] == 0:
        pred_list = list(pred_tuple)
        result1 = param_check()
        pred_list[2],pred_list[4] = pred_list[4],pred_list[2]
        pred_tuple = tuple(pred_list)
        result2 = param_check()
        pred_list[3],pred_list[5] = pred_list[5],pred_list[3]
        pred_tuple = tuple(pred_list)
        result3 = param_check()
        pred_list[2],pred_list[4] = pred_list[4],pred_list[2]
        pred_tuple = tuple(pred_list)
        result4 = param_check()
        return result1 or result2 or result3 or result4
    else:
        return param_check()


def accuracy_evaluation(self,predicted_sketchs, sketches,input_param,encoding_info):
    type_idx, type_batch_idx,type_token_idx, refer_idx,refer_batch_idx,refer_posi,primitive_param_batch,primitive_param_posi,constraint_param_batch,constraint_param_posi,batch_prim_param_GT, batch_const_param_GT,batch_const_param_type_GT,primitive_GT, constraint_GT, primitive_length, constraint_length = encoding_info
    parameter_collection,    structual_probability, refer_prob,structual_onehot_collection,cross_ref_detail = predicted_sketchs[0], predicted_sketchs[1].clone().detach() ,predicted_sketchs[2].clone().detach(),predicted_sketchs[3],predicted_sketchs[4] #[B,S1,L,E],  _ ,[B,S1,S2+1],  [B,P,S1,S1], [B,P,S1,1] P = 3
    ref_arg_in_prob,ref_arg_out_prob,ref_arg_in_crosscope_prob_reverse = cross_ref_detail
    structual_logprobability = torch.log(structual_probability + 1e-10)
    parameter_L0_output = parameter_collection[1]
    B,S0,E = parameter_L0_output.shape  #S1
    abstruction = self.max_abstruction
    type_total_num = 0
    type_right_num = torch.tensor([0],device=self.device)
    refer_total_num = 0
    refer_right_num = torch.tensor([0],device=self.device)
    non_type_total_num = 0
    non_type_right_num = torch.tensor([0],device=self.device)
    refer_inscope_num = torch.tensor([0],device=self.device)
    refer_inscope_correct_num = torch.tensor([0],device=self.device)



    ##check gluing reference in the same argument
    max_ptr_num = self.ref_out_argument_num      #the max number of the previous library
    new_arg_out_num = self.model.module.ref_out_argument_num
    Q = self.max_detection_query  #number of query
    L = self.max_abstruction # number of abstraction
    C = Q*L #candidate number

    ref_inscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,L,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_crossscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(L,new_arg_out_num,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
    ref_dia_idx_tensor = torch.ones_like(ref_arg_in_prob)
    ref_dia_idx_tensor[:,ref_inscope_dia_idx_not[0],:,ref_inscope_dia_idx_not[1]] = 0
    ref_dia_idx_tensor[:,ref_crossscope_dia_idx_not[0],:,C + ref_crossscope_dia_idx_not[1]] = 0
    ref_idx_not = torch.nonzero(ref_dia_idx_tensor,as_tuple=True)
    ref_arg_in_argmax = ref_arg_in_prob[ref_idx_not].reshape(B,Q,L,max_ptr_num,L+new_arg_out_num).argmax(4) #[B,Q,L,matptr,L*A+arg]
    ref_arg_in_argmax = torch.where(ref_arg_in_argmax<L,ref_arg_in_argmax,-ref_arg_in_argmax + L-1)

    


    #initialize reference masks
    L0_struc_prob = structual_probability #[B,S1,S2 + 1]
    L0_type_index = L0_struc_prob.argmax(dim=2) #[B,S1]


    param_GT_list = []
    param_list = []
    param_const_list = []
    param_const_type_list = []
    primitive_GT_list = []
    for i in range(len(primitive_length)):
        p_len = primitive_length[:i].sum()
        p_limit = primitive_length[i]
        c_len = constraint_length[:i].sum()
        c_limit = constraint_length[i]
        if self.constraint_param_flag:
            param_list.append(torch.cat([input_param[0][p_len:p_len+p_limit],input_param[1][c_len:c_len+c_limit]],dim=0))
        else:
            param_list.append(input_param[0][p_len:p_len+p_limit])
        param_GT_list.append(batch_prim_param_GT[p_len:p_len+p_limit])
        param_const_list.append(batch_const_param_GT[c_len:c_len+c_limit])
        param_const_type_list.append(batch_const_param_type_GT[c_len:c_len+c_limit])
        primitive_GT_list.append(primitive_GT[p_len:p_len+p_limit])

    ###
    pred_prim_count = 0
    pred_const_count = 0
    GT_prim_count = 0
    GT_const_count = 0
    prim_match_count = 0
    const_match_count = 0
    for batch_idx in range(len(sketches)): #B
        sketch = sketches[batch_idx]

        cost_matrix = self.build_up_costmatrix(sketches[batch_idx],structual_logprobability,batch_idx, L0_type_index[batch_idx], refer_prob,param_list[batch_idx],parameter_L0_output[batch_idx],param_GT_list[batch_idx],param_const_list[batch_idx],param_const_type_list[batch_idx])#[GT_S1,S1]
        

        #finish building the cost matrix, get linear assignment
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_idx_np, col_idx_np = linear_sum_assignment(cost_matrix_np)
        row_idx = torch.tensor(row_idx_np,dtype = torch.long, device=self.device)
        col_idx = torch.tensor(col_idx_np,dtype = torch.long, device=self.device)


        #### Calculate the F-score for primitive and constraint matching!
        GT_primitive_set = []
        pred_primitive_set = []
        current_prim_param = param_GT_list[batch_idx]
        current_prim_type = primitive_GT_list[batch_idx]
        for i,t in enumerate(current_prim_type.cpu().tolist()):
            GT_primitive_set.append(tuple([t] + current_prim_param[i][:self.primitive_length_mapping[t]].tolist()))

        if self.param_code_flag:
            primitive_param_decode,_ = self.model.module.param_encode_model(current_prim_param,L0_type_index[batch_idx],param_code = parameter_L0_output[batch_idx],encode_flag = False,primitive_flag =True)
        else:
            primitive_param_decode,_ = self.model.module.param_encode_model(current_prim_param,L0_type_index[batch_idx],param_code = parameter_L0_output[batch_idx],encode_flag = False,primitive_flag =True)
        primitive_param_result = primitive_param_decode.argmax(2)
        for i,pred_type in enumerate(L0_type_index[batch_idx].tolist()):
            if pred_type > 3:
                continue
            pred_primitive_set.append(tuple([pred_type] + primitive_param_result[i][:self.primitive_length_mapping[pred_type]].tolist()))

        ### calculate the union set
        union_primitive_list = []
        matched_GT = []
        for j,gt_tuple in enumerate(GT_primitive_set):
            for i,pred_tuple in enumerate(pred_primitive_set):
                if self.compare_primitive(gt_tuple,pred_tuple):
                    union_primitive_list.append(gt_tuple)
                    matched_GT.append(j)
                    break

        ###fill in the pred_constraint_set here!
        pred_constraint_set = []
        var_pred_counter = 0
        col_idx_list = col_idx.cpu().tolist()
        refer_point_matrix = refer_prob[batch_idx].argmax(2)
        for i,pred_type in enumerate(L0_type_index[batch_idx].tolist()):
            if (pred_type <= 3 or pred_type == 19):
                continue
            ###first translate the specific reference candidate
            refer_list = refer_point_matrix[i].tolist()
            matched_refer_list = []
            for j in refer_list:
                try:
                   a = col_idx_list.index(j)
                   if a  not in matched_GT:
                       a = -1
                except:
                    a = -1
                matched_refer_list.append(a)
            if pred_type in (10,14,15):
                a = matched_refer_list[0]
                pred_constraint_set.append(tuple([pred_type, matched_refer_list[0], matched_refer_list[1]]))
            elif pred_type in (6,8):
                pred_constraint_set.append(tuple([pred_type,matched_refer_list[0]]))
                pred_constraint_set.append(tuple([pred_type,matched_refer_list[0], matched_refer_list[1]]))
                var_pred_counter += 1
            else:
                pred_constraint_set.append(tuple([pred_type,matched_refer_list[0], matched_refer_list[1]]))

        GT_constraint_set = []
        entity_index_dict = {}
        for entity_idx, entity_key in enumerate(sketch.entities):
            entity_node = sketch.entities[entity_key]
            selected_idx = col_idx[entity_idx]   # the linear sum selection index of the candidate
            node_type = self.primitive_type_mapping_table[entity_node.type].item()
            entity_index_dict[entity_key] = entity_idx
            # primitive type loss
            entity_target_type = torch.tensor(node_type,dtype=torch.long,device = self.device,requires_grad=False)
            entity_pred_type = structual_probability[batch_idx, selected_idx].argmax()
            type_total_num += 1
            type_right_num += (entity_pred_type == entity_target_type)


        for constraint_idx, constraint_key in enumerate(sketch.constraints):
            constraint_idx = constraint_idx+ len(sketch.entities)
            constraints_node = sketch.constraints[constraint_key]
            selected_idx = col_idx[constraint_idx]   # the linear sum selection index of the candidate
            node_type = self.constraint_type_mapping_table[constraints_node.type]
            # primitive type loss
            entity_target_type = node_type
            entity_pred_type = structual_probability[batch_idx, selected_idx].argmax()
            type_total_num += 1
            type_right_num += (entity_pred_type == entity_target_type)



            
            GT_reference_index = []
            GT_reference_posi = []
            #enumerate through parameter to perform attributes loss first
            reference_count = 0

            for param in constraints_node.parameters:  #decode_result [B,S1,L,E]
                if isinstance(param, LocalReferenceParameter):
                    GT_reference_index.append(entity_index_dict[param.get_referenceMain()])
                    GT_reference_posi.append(reference_count)
                    reference_count += 1

            GT_constraint_set.append(tuple([node_type.item()]+GT_reference_index))


            #handle the reference here!
            ref_GT_idx = col_idx[torch.tensor(GT_reference_index,dtype = torch.long,device=self.device)]  #[R] 
            ref_GT_posi = torch.tensor(GT_reference_posi,dtype=torch.long,device=self.device)
            GT_reference_tensor =  ref_GT_idx
            pred_reference_tensor = refer_prob[batch_idx, selected_idx,ref_GT_posi].argmax(dim=1) # [R]
            refer_total_num += GT_reference_tensor.shape[0]
            refer_right_num += (GT_reference_tensor == pred_reference_tensor).sum() 

            selected_idx_group = torch.div(selected_idx, abstruction, rounding_mode= "trunc")
            pred_reference_group = torch.div(pred_reference_tensor, abstruction, rounding_mode= "trunc")
            refer_inscope_num += (selected_idx_group == pred_reference_group).sum()
            refer_inscope_correct_num += ((selected_idx_group == pred_reference_group) * (GT_reference_tensor == pred_reference_tensor)).sum()#[R]



        GT_constraint_set = set(GT_constraint_set)
        pred_constraint_set = set(pred_constraint_set)
        union_constraint_set = (GT_constraint_set & pred_constraint_set)

        pred_prim_count += len(pred_primitive_set)
        pred_const_count += (len(pred_constraint_set) -  var_pred_counter)
        GT_prim_count += len(GT_primitive_set)
        GT_const_count += len(GT_constraint_set)
        prim_match_count += len(union_primitive_list)
        const_match_count += len(union_constraint_set)

        #evaluate the non-type accuracy
        all_index = torch.ones(S0)
        all_index[col_idx] = 0
        unslected_index = all_index.nonzero()[:,0] #[U] # U is the index of unselected 
        none_type_target = torch.tensor(self.schema_length,dtype=torch.long,device=self.device).repeat(unslected_index.shape[0]) #[U]
        none_type_pred = structual_probability[batch_idx, unslected_index].argmax(dim = 1) #[U,S2+1] -> [U]
        non_type_total_num += unslected_index.shape[0]
        non_type_right_num += (none_type_target == none_type_pred).sum()


    return (type_total_num,type_right_num.detach().item()) ,\
        (refer_total_num, refer_right_num.detach().item(),refer_inscope_num.detach().item(),refer_inscope_correct_num.detach().item()),\
            (non_type_total_num, non_type_right_num.detach().item()), \
    (pred_prim_count, pred_const_count,GT_prim_count,GT_const_count,prim_match_count,const_match_count)


import torch
import torch.nn.functional as F
from torch import nn
from model_transformer import *
from model_embedding import *
class CADModel(nn.Module):
    def __init__(self,specs_arch,specs_embeded,rank, seed, specs):
        super(CADModel,self).__init__()
        torch.manual_seed(seed)
        self.detection_dim = specs["NetworkSpecs"]["detection_embedding_dim"]
        self.recon_dim = specs_arch["recon_embedding_dim"]
        self.key_embedding_dim = specs["EmbeddingStructure"]["key_dim"]
        self.nhead = specs_arch["atten_head_num"]
        self.moving_avg_library_update = specs["moving_avg_library_update"]["update_flag"]
        self.commitment_ratio = specs["Loss"]["commitment_ratio"]
        self.code_regulation = specs["code_regulation"]
        self.validity_ratio = specs["Loss"]["validity_ratio"]
        self.validity_flag = specs["validity_flag"]
        self.batch = 0
        self.epoch = 0
        # dead code retrival section:
        self.num_library = specs["EmbeddingStructure"]["num_library"]
        self.deadcode_retrival = specs["dead_code_retrival"]["flag"]
        self.deadcode_start_epoch = specs["dead_code_retrival"]["starting_epoch"]
        self.deadcode_batch_freq = specs["dead_code_retrival"]["batch_freq"]
        self.deadcode_stop_num = specs["dead_code_retrival"]["stop_num"]
        self.code_retrival_size = specs["dead_code_retrival"]["code_retrival_size"]
        self.code_retrival_threshold = specs["dead_code_retrival"]["code_retrival_threshold"]

        ##Architecture 
        self.normalize_before = specs_arch["normalize_before"]
        self.max_detection_query = specs["EmbeddingStructure"]["max_detection_query"]
        self.max_abstruction = specs_embeded["max_abstruction_decompose_query"]
        self.ref_in_argument_num = specs["EmbeddingStructure"]["ref_in_argument_num"]
        self.ref_out_argument_num = specs["EmbeddingStructure"]["ref_out_argument_num"]


        #self.query_validity_FC = FCBlock(2,2 ,self.detection_dim)
        self.device = torch.device(rank if torch.cuda.is_available() else "cpu")

        ### here we define the transformer encoder
        detection_encoder_layer = TransformerEncoderLayer(self.detection_dim, self.nhead, specs_arch["feedforward_dim"],
                                                specs_arch["dropout"], specs_arch["activation"], self.normalize_before)
        detection_encoder_norm = nn.LayerNorm(self.detection_dim) if self.normalize_before else None
        self.detection_encoder = TransformerEncoder(detection_encoder_layer, specs_arch["num_encoder_layers"], detection_encoder_norm)

        detection_decoder_layer = TransformerDecoderLayer(self.detection_dim, self.nhead, specs_arch["feedforward_dim"],
                                                specs_arch["dropout"], specs_arch["activation"], self.normalize_before)

        detection_decoder_norm = nn.LayerNorm(self.detection_dim)
        parameter_decoder_norm = nn.LayerNorm(self.detection_dim)
        detection_decoder_layer = TransformerDecoderLayer(self.detection_dim, self.nhead, specs_arch["feedforward_dim"],
                                                specs_arch["dropout"], specs_arch["activation"], self.normalize_before)
        self.detection_decoder = TransformerDecoder(detection_decoder_layer, specs_arch["num_detection_decoder_layers"], detection_decoder_norm,
                                          return_intermediate=False) # here return intermediate means return the intermediate feature from the decoder layers
        self.parameter_decoder = TransformerDecoder(detection_decoder_layer, specs_arch["num_detection_decoder_layers"], parameter_decoder_norm)
        ### these FCs are the bridge between structual and parameter decoder
        self.structual_FC = FCBlock(specs_arch["mid_fc_layers"],self.detection_dim,self.detection_dim)
        self.parameter_FC = FCBlock(specs_arch["mid_fc_layers"],self.detection_dim,self.detection_dim)

        #### for key gen embedding
        self.structual_expansion_FC = FCBlock(2,self.recon_dim*self.max_abstruction + (self.ref_in_argument_num  + self.ref_out_argument_num) * self.recon_dim,self.detection_dim)
        self.parameter_expansion_FC = FCBlock(2,self.detection_dim*self.max_abstruction,self.detection_dim)
        self.done_parameter_FC = FCBlock(3,self.detection_dim,self.detection_dim)

        #### for query positional embedding
        factor = (self.max_detection_query * self.ref_out_argument_num) *(self.max_detection_query * self.ref_in_argument_num)
        in_factor = self.ref_in_argument_num * self.max_abstruction
        out_factor = self.max_abstruction * self.ref_out_argument_num * (self.max_abstruction +self.ref_out_argument_num)
        self.lib_ref_prob_in_FC = FCBlock(specs_arch["reference_FC_layer"], in_factor,self.ref_in_argument_num * self.recon_dim)
        self.lib_ref_prob_out_FC = FCBlock(specs_arch["reference_FC_layer"], out_factor,self.ref_out_argument_num * self.recon_dim)
        self.query_gen_arg_FC = FCBlock(specs_arch["reference_FC_layer"], factor,self.detection_dim)  ##need to be fix here
        self.query_select_FC = FCBlock(specs_arch["type_FC_layer"], self.key_embedding_dim[-1],self.detection_dim)

        #attribute FC very short and simple!
        self.type_FC = FCBlock(specs_arch["type_FC_layer"], self.key_embedding_dim[0] ,self.recon_dim)
        self.query_validity_FC = FCBlock(specs_arch["reference_FC_layer"],2 ,self.detection_dim)
        #initialization all the network parameter except embedding
        self.initialize_model_weight()



    def initialize_model_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def parameter_expansion(self, parameter_input):
        parameter_input_code = parameter_input[0] 
        B,S1,E = parameter_input_code.shape 
        expand_factor = self.max_abstruction
        # for key gen embedding 
        parameter_decode = self.parameter_expansion_FC(parameter_input_code).reshape(B,S1,expand_factor,self.detection_dim).flatten(1,2) #[B,S1,L,E]
        return parameter_decode



    def structual_expansion(self,structual_selection_code):
        ##then process reference
        B,S1,E = structual_selection_code.shape
        expand_factor = self.max_abstruction
        structural_decoder_query = self.structual_expansion_FC(structual_selection_code) #[B,S1,L*E+A*E]
        structual_input_L0 = structural_decoder_query[:,:,:self.recon_dim * expand_factor].reshape(B,S1,expand_factor,self.recon_dim).flatten(1,2) #[B,S1,L*E+A*E] -> [B,S1,L*E] -> [B,S1,L,E] ->[B,S1*L,E]
        ref_arg_ptr_inscope_in = structural_decoder_query[:,:,self.recon_dim * expand_factor:self.recon_dim * (expand_factor + self.ref_in_argument_num)].reshape(B,S1,self.ref_in_argument_num,self.recon_dim)#[B,S1,L*E+A*E] -> [B,S1,A*E] ->[B,S1,A,E]
        ref_arg_ptr_inscope_out = structural_decoder_query[:,:,self.recon_dim * (expand_factor + self.ref_in_argument_num):].reshape(B,S1,self.ref_out_argument_num,self.recon_dim)#[B,S1,L*E+A*E] -> [B,S1,A*E] ->[B,S1,A,E]

        return (structual_input_L0,ref_arg_ptr_inscope_in,ref_arg_ptr_inscope_out)



    def quantitize(self, structual_input, cad_embedding, mode_flag):
        library_code = getattr(cad_embedding.module,"higher_library_embedding").to(self.device)
        B = structual_input.shape[0]
        batch_library = library_code.detach().unsqueeze(0).repeat(B,1,1) #[B,S2 + 1,E] where S2 is the total number of libary 

        #VQVAE euclidian distance calculation
        structual_type_code = self.query_select_FC(structual_input)  #
        batch_batch_library = batch_library.unsqueeze(1).repeat(1,structual_type_code.shape[1],1,1)
        batch_structual_type_code = structual_type_code.detach().unsqueeze(2).repeat(1,1,library_code.shape[0],1) #[B,S1,S2_1,E]
        batch_distance = (batch_batch_library - batch_structual_type_code).norm(dim=3) #[B,S1,S2_1]
        min_encoding_indices = batch_distance.argmin(dim=2).unsqueeze(2) #argmin 

        #### code selection#######
        structual_one_hot = torch.zeros_like(batch_distance, requires_grad=False)
        structual_one_hot.scatter_(2, min_encoding_indices,1)  #[B,S1,S2_1]
        structual_selection_code = torch.matmul(structual_one_hot,batch_library) #[B,S1,E]

        #### DEAD CODE RETRIVAL########
        unselected_index = torch.nonzero(cad_embedding.module.old_counting <= self.code_retrival_threshold, as_tuple = True)[0]
        if self.deadcode_retrival and self.batch % self.deadcode_batch_freq == 0 and mode_flag == "train" and unselected_index.shape[0] >=self.code_retrival_size: #and self.epoch >= self.deadcode_start_epoch:
            sample_index = unselected_index[torch.randperm(unselected_index.shape[0], device = self.device)[0:self.code_retrival_size]]
            selection_diff_norm = (structual_selection_code - structual_type_code).norm(dim=2)
            structual_type_code_index = torch.topk(selection_diff_norm.flatten().detach(), self.code_retrival_size,largest=True)
            structual_type_code_flatten = structual_type_code.flatten(0,1).detach()
            cad_embedding.module.scale_code[sample_index] = structual_type_code_flatten[structual_type_code_index[1]].detach()
            cad_embedding.module.mass[sample_index] = 1.0
            cad_embedding.module.higher_library_embedding[sample_index] = structual_type_code_flatten[structual_type_code_index[1]].detach()


        #the straight through tricks
        structual_selection_code = structual_type_code + (structual_selection_code - structual_type_code).detach()
        ###commitment loss 
        commitment_loss = F.mse_loss(structual_type_code,structual_selection_code.detach()) * self.commitment_ratio

        ###calculate the code ready for update
        structual_type_code_tile = structual_type_code.unsqueeze(2).repeat(1,1,self.num_library,1) # [B,S1,E] -> unsqueeze [B,S1,1,E] -> repeat [B,S1,S2+1,E]
        mask_structual_type_code_tile = torch.multiply(structual_type_code_tile,structual_one_hot.unsqueeze(dim=3))  # multiply [B,S1,S2+1,E] * [B,S1,S2+1] = [B,S1,S2+1,E]
        step_code = mask_structual_type_code_tile.sum(dim=(0,1)) # [B,S1,S2+1,E] sum-> [S2+1,E]
        selection_count = structual_one_hot.sum(dim=(0,1)).detach() #[B,S1,S2+1] sum -> [S2+1]
        return structual_one_hot, structual_selection_code,commitment_loss,step_code,selection_count


    def lower_layer_selection(self,structual_output,query_validity,validity_flag ,cad_embedding, mode_flag):
        structual_input_L0 = structual_output[0]

        ###preprocesss type selection and parameter selectino through FC 
        structual_type_code = self.type_FC(structual_input_L0)
        library_code = torch.cat([getattr(cad_embedding.module,"Lower_library_embedding").weight.to(self.device), getattr(cad_embedding.module,"Lower_library_nontype").weight.to(self.device)],dim=0) # cat [S2,E] ,[1,E] ->[S2+1,E]   
        structual_probability_raw = torch.einsum('bie,je->bij', structual_type_code,library_code)  #[B,S0,E]X[B,L0,E]  ->[B,S0,L0] 
        structual_one_hot = F.gumbel_softmax(structual_probability_raw,dim=2,hard=True) #[B,S1,L0] where L0 dimnesion is one-hot dimension 
        structual_probability_temp = torch.softmax(structual_probability_raw,dim=2)
        structual_probability = torch.zeros_like(structual_probability_temp)
        #structual_selection_code = torch.matmul(structual_one_hot,library_code)
        if validity_flag:
            structual_probability[:,:,:-1] = torch.multiply(structual_probability_temp[:,:,:-1] ,query_validity[:,:,0].repeat_interleave(self.max_abstruction,dim=1).unsqueeze(2))
            structual_probability[:,:,-1:] = torch.multiply(structual_probability_temp[:,:,:-1] ,query_validity[:,:,1].repeat_interleave(self.max_abstruction,dim=1).unsqueeze(2)).sum(2).unsqueeze(2) + structual_probability_temp[:,:,-1].unsqueeze(2)
        else:
            structual_probability = structual_probability_temp
        return structual_probability,structual_one_hot

    def reference_linkage(self,structual_output,structual_input):
        cross_linkage_info = structual_input[1]
        structual_input_L0,ref_arg_ptr_inscope_in,ref_arg_ptr_inscope_out = structual_output
        B = structual_input_L0.shape[0]
        
        ##decalre the necessary dims
        S1 = self.max_detection_query
        S1XAout = S1 * self.ref_out_argument_num  #number of receving argument out 
        S1XAin = S1 * self.ref_in_argument_num
        lib_abs = self.max_abstruction
        S0 = lib_abs * S1
        

        S1_dim = S1XAout * S1XAin
        cross_linkage_prod = self.query_gen_arg_FC(cross_linkage_info)
        ref_arg_out_prod = cross_linkage_prod[:,:,:S1_dim].reshape(B,S1,self.ref_out_argument_num,S1XAin).flatten(1,2)
        ref_out_idx = torch.ones(S1XAout , S1XAin ,device = self.device,requires_grad = False)
        ref_out_block_dia_idx = torch.nonzero(torch.block_diag(*[torch.ones(self.ref_out_argument_num,self.ref_in_argument_num,device = self.device) for _ in range(self.max_detection_query)]),as_tuple=True)
        ref_out_idx[ref_out_block_dia_idx] = 0
        ref_out_block_idx = torch.nonzero(ref_out_idx,as_tuple=True)
        prob_filling_out = torch.softmax(ref_arg_out_prod[:,ref_out_block_idx[0],ref_out_block_idx[1]].reshape(B,S1XAout,S1XAin -self.ref_in_argument_num),dim=2)   #get the block diagonal out [B,S1XA,S1XA]
        ######
        positive_entropic_loss = 0
        positive_entropic_loss += (prob_filling_out * torch.log(prob_filling_out+1e-7)).sum(2).mean()
        ######

        ref_arg_out_prob = torch.zeros_like(ref_arg_out_prod) #[B,S1XA,S1XA]
        ref_arg_out_prob[:,ref_out_block_idx[0],ref_out_block_idx[1]] = prob_filling_out.flatten(1,2) #[B,S1XA,S1XA]

        #dealing with inner ones
        ref_arg_in_prod = self.lib_ref_prob_out_FC(ref_arg_ptr_inscope_out.flatten(2,3)).reshape(B,S1,lib_abs,self.ref_out_argument_num,lib_abs + self.ref_out_argument_num)
        ref_inscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(lib_abs,lib_abs,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True)
        ref_crosscope_dia_idx_not = torch.nonzero(torch.block_diag(*[torch.ones(lib_abs,self.ref_out_argument_num,device = self.device) for _ in range(self.max_detection_query)]).bool().logical_not(),as_tuple=True) 
        ref_self_dia_idx = torch.nonzero(torch.block_diag(*[torch.ones(1,1,device = self.device) for _ in range(S0)]),as_tuple=True) 
        ref_arg_in_idx_tensor = torch.ones(B,S0,self.ref_out_argument_num,S0+S1XAout,device=self.device,requires_grad=False)
        ref_arg_in_idx_tensor[:,ref_self_dia_idx[0],:,ref_self_dia_idx[1]] = 0.0
        ref_arg_in_idx_tensor[:,ref_inscope_dia_idx_not[0],:,ref_inscope_dia_idx_not[1]] = 0.0
        ref_arg_in_idx_tensor[:,ref_crosscope_dia_idx_not[0],:,S0 + ref_crosscope_dia_idx_not[1]] = 0.0
        ref_mask_idx = torch.nonzero(ref_arg_in_idx_tensor,as_tuple=True)
        ref_arg_in_idx_tensor2 = torch.ones(lib_abs,lib_abs + self.ref_out_argument_num,device=self.device,requires_grad=False)
        ref_dia_idx_2 = torch.nonzero(torch.block_diag(*[torch.ones(1,1,device=self.device,requires_grad=False) for _ in range(lib_abs)]),as_tuple=True)
        ref_arg_in_idx_tensor2[ref_dia_idx_2]  = 0.0
        ref_mask_idx2 = torch.nonzero(ref_arg_in_idx_tensor2,as_tuple=True)
        prob_filling_in = torch.softmax(ref_arg_in_prod.transpose(2,3)[:,:,:,ref_mask_idx2[0],ref_mask_idx2[1]].reshape(B,S1,self.ref_out_argument_num,lib_abs,lib_abs+self.ref_out_argument_num-1),dim=4).transpose(2,3)#.transpose(2,3).flatten()   #[B,S1,L,3,L+A] -> softmax ->puting back[B,S1,L,3,L+A] ->flatten [BXS1XLX3XL+A]
        ref_inscope_prob = prob_filling_in.flatten(1,2)

        prob_filling_in = prob_filling_in.flatten()
        ref_arg_in_prob = torch.zeros_like(ref_arg_in_idx_tensor)   #[B,S0,3,S0+S1XA]
        ref_arg_in_prob[ref_mask_idx] = prob_filling_in #[B,S0,3,S0+S1XA]

        #dealing with cross-scope reference
        ref_arg_inscope_prod_in_filling = self.lib_ref_prob_in_FC(ref_arg_ptr_inscope_in.flatten(2,3)).reshape(B,S1,lib_abs ,self.ref_in_argument_num)
        prob_filling_in_crosscope_reverse = torch.softmax(ref_arg_inscope_prod_in_filling,dim=2).flatten(1,2)    # [B,S1,A,LXAin]  -> softmax ->flatten[B,S1XLXAin,A]

        ref_arg_in_crosscope_prob_reverse = torch.zeros(B,S0,S1XAin,device=self.device) #[B,S0,S1XA]
        ref_crosscope_dia_idx = torch.nonzero(torch.block_diag(*[torch.ones(lib_abs,self.ref_in_argument_num,device = self.device) for _ in range(self.max_detection_query)]).bool(),as_tuple=True) 
        ref_arg_in_crosscope_prob_reverse[:,ref_crosscope_dia_idx[0],ref_crosscope_dia_idx[1]] = prob_filling_in_crosscope_reverse.flatten(1,2) #[B,S0,S1XA]
        ref_arg_in_crosscope_prob = ref_arg_in_prob[:,:,:,S0:] #[B,S0,3,S0+S1XA] slicing-> [B,S0,3,S1XA]
        ref_arg_temp = torch.einsum('bijk,bkl->bijl', ref_arg_in_crosscope_prob, ref_arg_out_prob[:,:,:S1XAin]) #[B,S0,3,S1XA] X[B,S1XA,S1XA] -> [B,S0,3,S1XA]
        ref_arg_crosscope_prob = torch.einsum('bijk,blk->bijl', ref_arg_temp, ref_arg_in_crosscope_prob_reverse) #[B,S0,3,S1XA] X [B,S0,S1XA] -> #[B,S0,3,S0]

        #assign in-scope reference to its own position
        ref_arg_in_inscope_prob = ref_arg_in_prob[:,:,:,:S0] #[B,S0,3,S0+S1XA] slicing-> [B,S0,3,S0]
        ref_inscope_dia_idx = torch.nonzero(torch.block_diag(*[torch.ones(lib_abs,lib_abs,device = self.device) for _ in range(self.max_detection_query)]).bool(),as_tuple=True)
        ref_arg_crosscope_prob[:,ref_inscope_dia_idx[0],:,ref_inscope_dia_idx[1]] = ref_arg_in_inscope_prob[:,ref_inscope_dia_idx[0],:,ref_inscope_dia_idx[1]] #[B,S0,3,S0]
        refer_prob = ref_arg_crosscope_prob

        return refer_prob, (ref_arg_in_prob,ref_arg_out_prob,ref_arg_in_crosscope_prob_reverse),positive_entropic_loss,ref_inscope_prob

    def code_split(self,code):
        return [code[:,:self.max_detection_query],code[:,-1:]]

    def forward(self, input_tensors, cad_embedding,mode_flag):
        ##library expo to investigate the structure of the library
        if mode_flag == "library_expo":
            structual_output =  self.structual_expansion(input_tensors)
            structual_L0_probability,_ = self.lower_layer_selection(structual_output,None,False,cad_embedding, mode_flag)
            return structual_L0_probability

        batch_tensor, batch_posi, pad_mask = input_tensors
        B = batch_tensor.shape[1]
        detection_query = cad_embedding.module.detection_query_embed.weight.to(self.device).unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(detection_query,device=self.device)  #detection_query of shape [S1,B,E] S1 = max_detection_query B = batch E = embedding
        #detection encoder/decoder

        memory = self.detection_encoder(batch_tensor, src_key_padding_mask=pad_mask, pos=batch_posi) #memory [S0,B,E]
        detection_output = self.detection_decoder(tgt, memory, memory_key_padding_mask=pad_mask, query_pos=detection_query,pos=batch_posi)  #detection_output [S1,B,E]

        #Process FC accordingly and 
        detection_output = detection_output.transpose(0,1)  #batch first [B,S1,E]
        structual_input = self.structual_FC(detection_output) #[B,S1,E] S1: number of token detection decoder output E: embedding dimension output by FC
        parameter_input = self.parameter_FC(detection_output) #[B,S1,E]
        structual_input = self.code_split(structual_input)
        parameter_input = self.code_split(parameter_input)
        query_validity = torch.softmax(self.query_validity_FC(structual_input[0]),dim=2)
        validity_loss = -torch.log(query_validity[:,:,1]).mean() * self.validity_ratio

         
        structual_one_hot, structual_selection_code, commitment_loss,step_code,selection_count = self.quantitize(structual_input[0],cad_embedding, mode_flag)
        structual_output =  self.structual_expansion(structual_selection_code)
        parameter_L0_input =  self.parameter_expansion(parameter_input)
        structual_L0_probability,structual_L0_one_hot = self.lower_layer_selection(structual_output,query_validity,self.validity_flag,cad_embedding, mode_flag)

        refer_prob,cross_ref_detail,positive_entropic_loss,ref_inscope_prob = self.reference_linkage(structual_output,structual_input)

        structual_index = structual_L0_one_hot.argmax(2).flatten()
        structural_embedding = cad_embedding.module.type_embedding(structual_index).reshape(B,structual_L0_one_hot.shape[1],-1)
        parameter_decoder_input  = structural_embedding + parameter_L0_input

        parameter_decoder_input = parameter_decoder_input.transpose(0,1)
        tgt2 = torch.zeros_like(parameter_decoder_input,device=self.device)
        parameter_decoder_output = self.parameter_decoder(tgt2, memory, memory_key_padding_mask=pad_mask, query_pos=parameter_decoder_input,pos=batch_posi)

        parameter_decoder_output = parameter_decoder_output.transpose(0,1)
        parameter_decoder_output = self.done_parameter_FC(parameter_decoder_output)
        structual_onehot_collection = [structual_one_hot,structual_L0_one_hot]
        parameter_input_collection = [parameter_input[0],parameter_decoder_output]
        if mode_flag == "visualize":
            return (parameter_input_collection, structual_L0_probability,refer_prob,structual_onehot_collection,cross_ref_detail)
        elif mode_flag == "train":
            return (parameter_input_collection, structual_L0_probability,refer_prob, structual_onehot_collection, ref_inscope_prob, step_code, selection_count), (commitment_loss,validity_loss, positive_entropic_loss) # this probability is unlog and unnormalized, since it need to be processed by cross entropy, which already include log-softmax in it
        elif mode_flag == "validate":
            return (parameter_input_collection, structual_L0_probability,refer_prob,structual_onehot_collection,cross_ref_detail)



import networkx as nx
import json
import pickle
from sketchgraphs.data import  flat_array
import numpy as np
from sketchgraphs.data._entity import Arc, Circle, Line, Point
import sketchgraphs.data as datalib
import os
import argparse

def get_line_position(entity):
    startx,starty = entity.start_point
    endx,endy =  entity.end_point
    return (startx +endx)/2, (starty+endy) /2
def get_arc_position(entity):
    return entity.xCenter,entity.yCenter

def get_circle_position(entity):
    return entity.xCenter,entity.yCenter

def get_point_position(entity):
    return entity.x,entity.y
get_position = {
    Line : get_line_position,
    Arc : get_arc_position,
    Circle : get_circle_position,
    Point : get_point_position
}
color_name = [
    "#2b3d54", #blue
    "#ffff00", #yellow
    "#00ff1e", #green
    "#dd4b39", #red
]
color_name = [
    "blue", #blue
    "purple", #yellow
    "green", #green
    "red", #red
    "pink",
    "brown"
]
type_name =[
      "Line",
      "Point",
      "Circle",
      "Arc",
      "Coinc.",
      "Distance",
      "Horizon",
      "Parallel",
      "Vertical",
      "Tangent",
      "Length",
      "Perpend",
      "Midpoint",
      "Equal",
      "Diameter",
      "Radius",
      "Angle",
      "Concentric",
      "Normal"
]


def get_argsparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', "-d",required=True,
                        help='Path to training dataset')\
    
    parser.add_argument('--train_file', required=True,
                        help='The filename of the training data')\

    parser.add_argument('--experiment', "-e",required=True,
                        help='Path to experiment directory')\

    parser.add_argument('--epoch',required=True,
                        help='The epoch of result')
    return parser

def load_experiment_specification(experiment_directory):
    filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file ".format(experiment_directory)
            + '"specs.json"'
        )

    return json.load(open(filename))

if __name__ == "__main__":
    ###load argument and spec
    parser = get_argsparser()
    args = vars(parser.parse_args())
    specs = load_experiment_specification(args["experiment"])

    ### load also parameters
    AbstractionLevel = specs["EmbeddingStructure"]["max_abstruction_decompose_query"]
    sketch_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch.npy')
    sketch_data = flat_array.load_dictionary_flat(sketch_data_path) 
    stat_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch_stat.npz')
    stat_data = np.load(stat_data_path, allow_pickle = True) 
    group_type = stat_data['type']
    group_belong = stat_data['belong']
    group_corr = stat_data['corr']
    graph_data = stat_data['graph']
    ref_in_argument_num = specs["EmbeddingStructure"]["ref_in_argument_num"]
    ref_out_argument_num = specs["EmbeddingStructure"]["ref_out_argument_num"]


    out_arg_base = 100 #hardcode large number
    in_arg_base = 200
    penwidth = 3
    fontsize = 30
    for sketch_idx,g_data in enumerate(graph_data):
        H = g_data
        t_ = group_type[sketch_idx]
        b_ = group_belong[sketch_idx]
        sketch = sketch_data["sequences"][sketch_idx]
        entity_key = list(sketch.entities.keys())
        clusters = [[],[],[],[],[],[]]
        
        for i,key in enumerate(H.nodes):
            if key <= -key-out_arg_base and key >= -in_arg_base:
                node = H.nodes[key]
                node["color"] = color_name[int((-key-out_arg_base)/ref_out_argument_num)]
                node["label"] = "Out arg: " + str((key % ref_out_argument_num))
                clusters[int((-key-out_arg_base)/ref_out_argument_num)].append(key)
                node["penwidth"] = penwidth
                node["fontsize"] = fontsize
                continue  
            if key <= -in_arg_base:
                node = H.nodes[key]
                node["color"] = color_name[int((-key-in_arg_base)/ref_in_argument_num)]
                node["label"] = "In arg: " + str((key % ref_in_argument_num))
                clusters[int((-key-in_arg_base)/ref_in_argument_num)].append(key)
                node["penwidth"] = penwidth
                node["fontsize"] = fontsize
                continue     
            node = H.nodes[key]
            node["title"] = str(node["predicted_idx"])
            node["label"] = type_name[node["type"]]
            query_num = int(node["predicted_idx"]/AbstractionLevel)
            node["color"] = color_name[query_num]
            node["penwidth"] = penwidth
            node["fontsize"] = fontsize
            clusters[query_num].append(key)


        edg = []
        repeated_edge = []


        for key in H.edges:
            edge = H.edges[key]
            if edge["predicted_idx"] == -5:
                edge["color"] = "black"
                edge["penwidth"] = penwidth
                edge["fontsize"] = fontsize
                continue
            #edge["label"] = type_name[edge["type"]]
            edge["color"] = color_name[int(edge["predicted_idx"]/AbstractionLevel)]
            edge["penwidth"] = penwidth
            edge["fontsize"] = fontsize
            reverse_edge = list(key)
            reverse_edge[0],reverse_edge[1] = reverse_edge[1],reverse_edge[0]
            if reverse_edge[0] == reverse_edge[1]:
                if key[:2] not in repeated_edge:
                    repeated_edge.append(tuple(reverse_edge[:2]))
                    continue
            else:
                if reverse_edge in H.edges and key not in repeated_edge:
                    repeated_edge.append(tuple(reverse_edge))
                    edge["dir"] = "both"



        for e in repeated_edge:
           H.remove_edge(*e)


        for key in H.edges:
            edge = H.edges[key]
            if edge["predicted_idx"] == -5:
                continue
            if not ( key[0] < 0 and key[1]>= 0):
                edge["label"] = type_name[edge["type"]]

        G = nx.drawing.nx_agraph.to_agraph(H)

        for i,cluster in enumerate(clusters):
            B = G.add_subgraph(cluster, name="cluster_" + str(i+1), rank="same")
        B.graph_attr["rank"] = "same"

        arg = '-Grankdir="LR" -Gnodesep="0.2" -Gsize="40,20" -Nshape="rectangle" -Nstyle="rounded,bold"'
        G.layout("dot",args=arg)
        G.layout("dot")
        outputdir = "./"
        G.draw(os.path.join(outputdir,"sketch"+str(sketch_idx) +".svg"))
        G.draw(os.path.join(outputdir,"sketch"+str(sketch_idx) +".png"))
        exit(1)





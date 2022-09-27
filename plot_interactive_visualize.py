from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
from sketchgraphs.data._constraint import ConstraintType, LocalReferenceParameter
from sketchgraphs.data import  flat_array
import sketchgraphs.data as datalib
import math
from matplotlib.backend_bases import MouseButton
import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
from sketchgraphs.data._entity import EntityType
from sketchgraphs.data._entity import Arc, Circle, Line, Point
from sketchgraphs.data.sequence import sketch_to_sequence, pgvgraph_from_sequence
from sketchgraphs_models import graph
import json
import os
import argparse
import re

lwlight = 2
texts_collection = []
#initialize color
color_name = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

constraint_name_filter = ["Line","Point","Circle","Arc","Constraint","Distance",
      "Horizontal","Parallel","Vertical","Tangent","Length","Perpendicular","Midpoint","Equal","Diameter",
      "Radius","Angle","Concentric","Normal"]
def get_cmap(n, name='plasma'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def string_strip(string):
    return re.findall(r"[-+]?\d*\.\d+|\d+", string)[0]

def _get_linestyle(entity):
    return '--' if entity.isConstruction else '-'

###Some helper function
def sketch_point(ax,point: Point,   color='black', show_subnodes=False, high_light = False):
    if high_light:
        s = 125
    else:
        s = 35
    if high_light and (math.isnan(point.x) or math.isnan(point.y)):
        text = ax.title._text + "invalid parameter not draw"
        ax.set_title(text)
    ax.scatter(point.x, point.y, color=color,s = s, marker='.')

def sketch_line(ax, line: Line, color='black', show_subnodes=False, high_light = False):
    start_x, start_y = line.start_point
    end_x, end_y = line.end_point
    if high_light:
        lw = 4

    else:
        lw = lwlight

    if high_light and (math.isnan(start_x) or math.isnan(start_y) or math.isnan(end_x) or math.isnan(end_y)):
        text = ax.title._text + "invalid parameter not draw"
        ax.set_title(text)

    if show_subnodes:
        marker = '.'
    else:
        marker = None
    ax.plot((start_x, end_x), (start_y, end_y), color = color ,linestyle=_get_linestyle(line), marker=marker, lw = lw)

def sketch_circle(ax, circle: Circle,  color='black', show_subnodes=False, high_light = False):
    if high_light:
        lw = 4

    else:
        lw = lwlight

    if high_light and (math.isnan(circle.xCenter) or math.isnan(circle.yCenter)or math.isnan(circle.radius)):
        text = ax.title._text + "invalid parameter not draw"
        ax.set_title(text)
    patch = matplotlib.patches.Circle(
        (circle.xCenter, circle.yCenter), circle.radius,
        fill=False, linestyle=_get_linestyle(circle), color=color, lw = lw)
    if show_subnodes:
        ax.scatter(circle.xCenter, circle.yCenter, c=color, marker='.', zorder=20)
    ax.add_patch(patch)

def sketch_arc(ax, arc: Arc,  color='black', show_subnodes=False, high_light = False):
    angle = math.atan2(arc.yDir, arc.xDir) * 180 / math.pi
    startParam = arc.startParam * 180 / math.pi
    endParam = arc.endParam * 180 / math.pi
    if high_light:
        lw = 4
    else:
        lw = lwlight

    if high_light and (math.isnan(arc.xCenter) or math.isnan(arc.yCenter)or math.isnan(arc.radius)or math.isnan(angle)or math.isnan(startParam)or math.isnan(endParam)):
        text = ax.title._text + "invalid parameter not draw"
        ax.set_title(text)

    if arc.clockwise:
        startParam, endParam = -endParam, -startParam

    ax.add_patch(
        matplotlib.patches.Arc(
            (arc.xCenter, arc.yCenter), 2*arc.radius, 2*arc.radius,
            angle=angle, theta1=startParam, theta2=endParam,
            linestyle=_get_linestyle(arc), color=color, lw = lw))

    if show_subnodes:
        ax.scatter(arc.xCenter, arc.yCenter, c=color, marker='.')
        ax.scatter(*arc.start_point, c=color, marker='.', zorder=40)
        ax.scatter(*arc.end_point, c=color, marker='.', zorder=40)


_PLOT_BY_TYPE = {
    Arc: sketch_arc,
    Circle: sketch_circle,
    Line: sketch_line,
    Point: sketch_point
}

def render_sketch(sketch, high_light_entity = [],color_flag = False,group_type = None, group_belong = None, save_file= None,ax=None, rescale = True,show_axes=False, show_origin=False, hand_drawn=False, show_subnodes=False):
    """Renders the given sketch using matplotlib.
    Parameters
    ----------
    sketch : Sketch
        The sketch instance to render
    ax : matplotlib.Axis, optional
        Axis object on which to render the sketch. If None, a new figure is created.
    show_axes : bool
        Indicates whether axis lines should be drawn
    show_origin : bool
        Indicates whether origin point should be drawn
    hand_drawn : bool
        Indicates whether to emulate a hand-drawn appearance
    show_subnodes : bool
        Indicates whether endpoints/centerpoints should be drawn
    Returns
    -------
    matplotlib.Figure
        If `ax` is not provided, the newly created figure. Otherwise, `None`.
    """

    if hand_drawn:
        saved_rc = mpl.rcParams.copy()
        plt.xkcd(scale=1, length=100, randomness=3)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        fig = None

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    if not show_axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        _ = [line.set_marker('None') for line in ax.get_xticklines()]
        _ = [line.set_marker('None') for line in ax.get_yticklines()]

        # Eliminate lower and left axes
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')

    if show_origin:
        point_size = mpl.rcParams['lines.markersize'] * 1
        ax.scatter(0, 0, s=point_size, c='black')
    

    for idx,entity in enumerate(sketch.entities.values()):
        sketch_fn = _PLOT_BY_TYPE.get(type(entity))
        entity_key = list(sketch.entities.keys())[idx]
        if len(high_light_entity) !=0  and entity_key in high_light_entity:
            high_light_flag = True
        else:
            high_light_flag = False
        

        if sketch_fn is None:
            continue
        color = 'black'

        if color_flag == True:
            belong = group_belong[idx]
            color_idx = int(belong/AbstractionLevel)
            if isinstance(color_buffer,list):
                color = color_buffer[color_idx]
            else:
                color = color_buffer(color_idx)

        sketch_fn(ax, entity, color= color, show_subnodes=show_subnodes, high_light = high_light_flag)

    if hand_drawn:
        mpl.rcParams.update(saved_rc)

    plt.draw()
    return None


def click_region_detection(sketch, event):
    double_click_flag = True if event.dblclick else False
    for entity_key in sketch.entities.keys():
        stroke = sketch.entities[entity_key]
        if type(stroke) == Line:
            start_x, start_y = stroke.start_point
            end_x, end_y = stroke.end_point
            length = math.sqrt((start_y - end_y)**2 + (start_x - end_x)**2)
            a = (start_x - end_x)
            if abs(a) < 1e-5:
                a = 1e-5
            a = (start_y - end_y)/a #y=ax+b
            b = start_y - a* start_x
            distance = abs(event.ydata - a* event.xdata - b)/math.sqrt(a*a+b*b)
            if distance < 0.05 and length > math.sqrt((start_y - event.ydata)**2 + (start_x - event.xdata)**2) and length > math.sqrt((end_y - event.ydata)**2 + (end_x - event.xdata)**2):
                if double_click_flag:
                    double_click_flag = False
                else:
                    return entity_key
        if type(stroke) == Circle:
            distance = math.sqrt((stroke.yCenter - event.ydata)**2 + (stroke.xCenter -  event.xdata)**2)
            if abs(distance - stroke.radius)<0.025:
                if double_click_flag:
                    double_click_flag = False
                else:
                    return entity_key
        
        if type(stroke) == Point:
            distance = math.sqrt((stroke.y - event.ydata)**2 + (stroke.x -  event.xdata)**2)
            if abs(distance)<0.025:
                if double_click_flag:
                    double_click_flag = False
                else:
                    return entity_key
        
        if type(stroke) == Arc:
            distance = math.sqrt((stroke.yCenter - event.ydata)**2 + (stroke.xCenter -  event.xdata)**2)
            startParam = stroke.startParam * 180 / math.pi
            endParam = stroke.endParam * 180 / math.pi
            theta = math.atan2(event.ydata - stroke.yCenter,event.xdata - stroke.xCenter) * 180 /math.pi
            if theta<0:
                theta+= 360
            if startParam<0:
                startParam += 360
            if endParam<0:
                endParam += 360
            if startParam< endParam:  # two scenario arc pass 0 theta point,(positive X), or NOT
                if abs(distance - stroke.radius)<0.025 and startParam < theta and endParam > theta:
                    if double_click_flag:
                        double_click_flag = False
                    else:
                        return entity_key
            else:
                if abs(distance - stroke.radius)<0.025 and (startParam < theta or endParam > theta):
                    if double_click_flag:
                        double_click_flag = False
                    else:
                        return entity_key
    return None

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

    ##load parameter
    AbstractionLevel = specs["EmbeddingStructure"]["max_abstruction_decompose_query"]
    query_num = specs["EmbeddingStructure"]["max_detection_query"]
    total_L0 = AbstractionLevel * query_num



    ###load data loading
    sketch_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch.npy')
    sketch_data = flat_array.load_dictionary_flat(sketch_data_path) 
    stat_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch_stat.npz')
    stat_data = np.load(stat_data_path, allow_pickle = True) 
    GT_data_path = os.path.join(args["dataset_dir"],args["train_file"])   
    GT_sketch_data = flat_array.load_dictionary_flat(GT_data_path) 
    GT_data_corr_path = os.path.join(args["dataset_dir"],args["train_file"][:-4]+"_filter.npy")
    GT_sketch_corr_data = np.load(GT_data_corr_path) 

    stat_type = stat_data['type']
    stat_belong = stat_data['belong']
    stat_corr = stat_data['corr']

    color_buffer = [
    "blue", #blue
    "purple", #yellow
    "green", #green
    "red", #red
    "pink",
    "brown",
    "yellow",
    "#d62728", #8
    '#bbbd67', 
    '#4b818c',
    '#7f7f7f', 
    '#bcbd22'
]

    for sketch_idx in range(0,30):

        ###Load the raw data
        sketch = sketch_data["sequences"][sketch_idx]
        GT_correspond_idx = GT_sketch_corr_data[sketch_idx]
        sketch_GT = GT_sketch_data["sequences"][GT_correspond_idx]
        corr = stat_corr[sketch_idx].tolist()
        ttype = stat_type[sketch_idx]
        belong = stat_belong[sketch_idx]

        ###setting up the interface
        plt.rcParams["figure.figsize"] = (12,4.8)
        axcolor = 'lightgoldenrodyellow'

        fig, ax = plt.subplots(1, 2,sharex=True)
        bax = ax
        ax[0].set_position([0.17,0.1, 0.40, 0.80])
        ax[1].set_position([0.59,0.1, 0.40, 0.80])
        ax[0].set_aspect('equal', 'box')
        ax[1].set_aspect('equal', 'box')
        rax = plt.axes([0.00, 0.1, 0.07, 0.90], facecolor=axcolor, frameon=True)
        rax_GT = plt.axes([0.0725, 0.1, 0.07, 0.90], facecolor=axcolor, frameon=True)
        constraint_key = ["None"] + list(sketch.constraints.keys())
        constraint_key = [key[key.find("_")+1:]for key in constraint_key]
        constraint_key_GT = ["None"] + list(sketch_GT.constraints.keys())
        constraint_key_GT = [key[key.find("_")+1:]for key in constraint_key_GT]
        radio = RadioButtons(rax, constraint_key)
        radio_GT = RadioButtons(rax_GT, constraint_key_GT)

        ##Process the drawing
        pred_label_p = "None"
        GT_label_p = "None"
        def pred_constraint_reaction(pred_label):
            global pred_label_p
            pred_label_p = pred_label
            if GT_label_p != "None":
               radio_GT.set_active(0)
            ax[0].clear()
            ax[1].clear()
            reference_entity = []
            reference_GT_entity = [] 
            if pred_label != "None":
                idx = len(sketch.entities.keys()) + int(pred_label[pred_label.find("_")+1:])-1 
                pred_text = "Query: " + str(int(belong[idx]/AbstractionLevel))   +"| Library: " + str(int(ttype[idx][1]))+" |" 
                pred_text += str(sketch.constraints["c_"+pred_label].constraint_type)[15:]
                reference_entity = [param.value for param in sketch.constraints["c_"+pred_label].parameters if type(param) == LocalReferenceParameter]


                ##prepare for the GT matching
                GT_index = corr[idx]
                if GT_index == -1:
                    GT_text = "this pred does Not map to a GT"
                    GT_key= None
                else:
                    if GT_index >= len(sketch_GT.entities):
                        GT_key = list(sketch_GT.constraints.keys())[GT_index - len(sketch_GT.entities)]
                        GT_text = str(sketch_GT.constraints[GT_key].constraint_type)[15:] 
                        reference_GT_entity = [string_strip(param.value) for param in sketch_GT.constraints[GT_key].parameters if type(param) == LocalReferenceParameter]
                    else:
                        GT_key = list(sketch_GT.entities.keys())[GT_index] 
                        GT_text = str(type(sketch_GT.entities[GT_key]))[34:-2] + "map to entity"
                        reference_GT_entity = [GT_key]
                ax[0].set_title(pred_text)
                ax[1].set_title(GT_text)
           
            render_sketch(sketch,high_light_entity = reference_entity,ax = ax[0],color_flag = True,  group_type = ttype, group_belong = belong,rescale = True,show_axes = False)
            render_sketch(sketch_GT,ax = ax[1],high_light_entity = reference_GT_entity, color_flag = False, rescale = True,show_axes = False)


        def GT_constraint_reaction(GT_label):
            global GT_label_p
            GT_label_p = GT_label
            ax[0].clear()
            ax[1].clear()
            if pred_label_p != "None":
               radio.set_active(0)
            GTreference_entity = []
            reference_entity = []
            if GT_label != "None" :
                GTreference_entity = [string_strip(param.value) for param in sketch_GT.constraints["c_"+GT_label].parameters if type(param) == LocalReferenceParameter]
                GT_idx = len(sketch_GT.entities) + list(sketch_GT.constraints.keys()).index("c_"+GT_label)
                try: 
                    constraint_index = corr.index(GT_idx)
                except ValueError:
                    constraint_index = -1

                if constraint_index == -1:
                    pred_text = "this GT map to a non type candidate"
                else:
                    if constraint_index < len(sketch.entities):
                        pred_entitiy_key = list(sketch.entities.keys())[constraint_index]
                        type_text = str(type(sketch.entities[pred_entitiy_key]))[34:-2]
                        reference_entity = [pred_entitiy_key]
                    else:
                        pred_entitiy_key = list(sketch.constraints.keys())[constraint_index - len(sketch.entities)] 
                        type_text = str(sketch.constraints[pred_entitiy_key].constraint_type)[15:]
                        reference_entity = [param.value for param in sketch.constraints[pred_entitiy_key].parameters if type(param) == LocalReferenceParameter]
                    pred_text = "Query: " + str(int(belong[constraint_index]/AbstractionLevel))   +"| Library: " + str(int(ttype[constraint_index][1]))+type_text
                ax[0].set_title(pred_text)
                GT_type_text = str(sketch_GT.constraints["c_"+GT_label].constraint_type)[15:]
                ax[1].set_title(GT_type_text)
            render_sketch(sketch,high_light_entity = reference_entity,ax = ax[0],color_flag = True,  group_type = ttype, group_belong = belong,rescale = True,show_axes = False)
            render_sketch(sketch_GT,high_light_entity = GTreference_entity,ax = ax[1],color_flag = False, rescale = True,show_axes = False)


        radio.on_clicked(pred_constraint_reaction)
        rpos = rax.get_position().get_points()
        rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (0.5)
        for circ in radio.circles:
            circ.set_radius(0.08)
            circ.height /= rscale
        
        radio_GT.on_clicked(GT_constraint_reaction)
        rpos = rax_GT.get_position().get_points()
        rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (0.5)
        for circ in radio_GT.circles:
            circ.set_radius(0.08)
            circ.height /= rscale

        def on_click(event):
            #predicted click
            if event.button is MouseButton.LEFT and event.xdata!=None and event.ydata!= None and (event.inaxes == ax[0] or event.inaxes == ax[1]):
                if pred_label_p != "None":
                    radio.set_active(0)
                if GT_label_p != "None":
                    radio_GT.set_active(0)
            GT_key = None
            if event.button is MouseButton.LEFT and event.xdata!=None and event.ydata!= None and event.inaxes == ax[0]:
                ax[1].clear()
                ax[0].clear()
                click_return = click_region_detection(sketch, event)   ##this detect entity only 
                if click_return != None:
                    idx = int(click_return)
                    ##prepare for the GT matching
                    GT_index = corr[idx]
                    if GT_index == -1:
                        GT_text = "this pred does Not map to a GT"
                        GT_key= None
                    else:
                        if GT_index >= len(sketch_GT.entities):
                            GT_key = list(sketch_GT.constraints.keys())[GT_index - len(sketch_GT.entities)]
                            GT_text = str(sketch_GT.constraints[GT_key].constraint_type)[15:] + "map to constraint"
                        else:
                            GT_key = list(sketch_GT.entities.keys())[GT_index] 
                            GT_text = str(type(sketch_GT.entities[GT_key]))[34:-2] 
                    

                    pred_text = "Query: " + str(int(belong[idx]/AbstractionLevel))   +"|" + " Library: " + str(int(ttype[idx][1]))
                    ax[1].set_title(GT_text)
                    ax[0].set_title(pred_text)


                render_sketch(sketch,high_light_entity = [click_return],ax = ax[0],color_flag = True,  group_type = ttype, group_belong = belong,rescale = True,show_axes = True)
                render_sketch(sketch_GT,ax = ax[1],high_light_entity = [GT_key],color_flag = False, rescale = True,show_axes = True)
            
            # GT click
            if event.button is MouseButton.LEFT and event.xdata!=None and event.ydata!= None and event.inaxes == ax[1]:
                ax[1].clear()
                ax[0].clear()
                click_return = click_region_detection(sketch_GT, event)
                entitiy_key = None
                if click_return != None:
                    GT_idx = list(sketch_GT.entities.keys()).index(click_return)
                    try: 
                        entity_index = corr.index(GT_idx)
                    except ValueError:
                        entity_index = -1
                    else:
                        if entity_index >= len(sketch.entities.keys()):
                            entitiy_key = list(sketch.constraints.keys())[entity_index - len(sketch.entities)]
                        else:
                            entitiy_key = list(sketch.entities.keys())[entity_index] 

                    
                    if entity_index != -1:
                        pred_text = "Query: " + str(int(belong[entity_index]/AbstractionLevel))   +"|" + " Library: " + str(int(ttype[entity_index][1]))
                    else:
                        pred_text = "this GT map to a non type candidate"
                    GT_text = str(type(sketch_GT.entities[click_return]))[34:-2]
                    ax[1].set_title(GT_text)
                    ax[0].set_title(pred_text)
                render_sketch(sketch_GT,high_light_entity = [click_return],ax = ax[1],color_flag = False, rescale = True,show_axes = True)
                render_sketch(sketch,high_light_entity = [entitiy_key],ax = ax[0],color_flag = True,  group_type = ttype, group_belong = belong,rescale = True,show_axes = True)
        
        render_sketch(sketch_GT,ax = ax[1],color_flag = False, rescale = True,show_axes = False)
        render_sketch(sketch,ax = ax[0],color_flag = True,  group_type = ttype, group_belong = belong,rescale = True,show_axes = False)
        plt.connect('button_press_event', on_click)
        plt.show()


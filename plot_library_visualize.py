from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
from numpy.core.fromnumeric import size
from sketchgraphs.data._constraint import ConstraintType, LocalReferenceParameter
from sketchgraphs.data import  flat_array
import sketchgraphs.data as datalib
import math
from matplotlib.backend_bases import MouseButton
import json
import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
from sketchgraphs.data._entity import EntityType
from sketchgraphs.data._entity import Arc, Circle, Line, Point
from sketchgraphs.data.sequence import sketch_to_sequence, pgvgraph_from_sequence
from sketchgraphs_models import graph
import re
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
from matplotlib.widgets import AxesWidget, RadioButtons
import os
import matplotlib.pyplot as plt
import networkx as nx
import argparse


class MyRadioButtons(RadioButtons):

    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([],[], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(5)
        self.cnt = 0
        self.observers = {}





texts_collection = []
#initialize color
color_name = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def string_strip(string):
    return re.findall(r"[-+]?\d*\.\d+|\d+", string)[0]

def _get_linestyle(entity):
    return '--' if entity.isConstruction else '-'

def sketch_point(ax,point: Point,   color='black', show_subnodes=False, high_light = False):
    if high_light:
        s = 125
    else:
        s = 40
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
        lw = 1

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
        lw = 1

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
        lw = 1

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

def render_sketch(sketch, high_light_entity = [],color_list = None,group_type = None, group_belong = None, save_file= None,ax=None, rescale = True,show_axes=False, show_origin=False, hand_drawn=False, show_subnodes=False):
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
        if color_list[idx] != -1:
            color = color_buffer(color_list[idx])
        sketch_fn(ax, entity, color= color, show_subnodes=show_subnodes, high_light = high_light_flag)

    if hand_drawn:
        mpl.rcParams.update(saved_rc)
    if save_file != None:
        fig.savefig(save_file)
        plt.close(fig)
    else:
        plt.draw()
    return None

index_to_type = {
    0:"Line",
    1:"Point",
    2:"Circle",
    3:"Arc",
    4:"Coinci",
    5:"Distan",
    6:"Horizon",
    7:"Parall",
    8:"Vertica",
    9:"Tangent",
    10:"Length",
    11:"Perpend",
    12:"Mipoint",
    13:"Equal",
    14:"Diam",
    15:"Radius",  
    16:"Angle",  
    17:"Concen",
    18:"Normal",
}

common_ref_num = {
    4:2,
    5:2,
    6:2,
    7:2,
    8:2,
    9:2,
    10:1,
    11:2,
    12:2,
    13:2,
    14:1,
    15:1,  
    16:1,  
    17:2,
    18:2,
}
edge_map = {
    0:"black",
    1:"blue",
    2:"purple"
}


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
    ref_in_argument_num = specs["EmbeddingStructure"]["ref_in_argument_num"]
    ref_out_argument_num = specs["EmbeddingStructure"]["ref_out_argument_num"]
    query_num = specs["EmbeddingStructure"]["max_detection_query"]
    total_L0 = AbstractionLevel * query_num
    

    ###loading library
    node_map = []
    sketch_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch.npy')
    sketch_data = flat_array.load_dictionary_flat(sketch_data_path) 
    GT_data_path = os.path.join(args["dataset_dir"],args["train_file"])   
    GT_sketch_data = flat_array.load_dictionary_flat(GT_data_path) 
    GT_data_corr_path = os.path.join(args["dataset_dir"],args["train_file"][:-4]+"_filter.npy")
    GT_sketch_corr_data = np.load(GT_data_corr_path) 
    stat_data_path = os.path.join(args["experiment"],"result","sketch_visualization",specs["exp_name"]+'_epoch_'+args["epoch"]+'_sketch_stat.npz')
    stat_data = np.load(stat_data_path, allow_pickle = True) 
    lib_pointer = stat_data["pointer"]
    lib_argument = stat_data["arguement"]
    lib_structure = stat_data["structure"]
    group_type = stat_data['type']
    group_belong = stat_data['belong']
    group_corr = stat_data['corr']
    lib_select = stat_data['lib_select']




    color_buffer =get_cmap(10) 
    lib_num = len(specs["EmbeddingStructure"]["type_schema"])


    lib_gl = 0
    lib_level = 1
    data_size = len(sketch_data["sequences"])
    plt.rcParams["figure.figsize"] = (9,9)
    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            ax[i,j].set_aspect(1.0)
    fig.tight_layout()
    plt.subplots_adjust( wspace=0.2, 
                        hspace=0.1,
                        top = 0.87,
                        bottom=0.05)

    def text_reaction(expression):
        global lib_gl
        global radio
        try:
            interested_library = int(expression) 
        except:
            return
        lib_gl  = interested_library

        print("library",interested_library,":",lib_structure[interested_library])

        L0_list = list(lib_structure[interested_library])
        labels = []
        for num,i in enumerate(L0_list):
            if i <= 18:
               labels.append(str(num)+":"+index_to_type[i])
            elif i == lib_num:
               labels.append(str(num)+":None")
            else:
               labels.append(str(num)+":lib" + str(i))
        radio =  MyRadioButtons(rax ,labels, ncol=4)
        radio.connect_event('pick_event', radio_reaction)  
        draw_library(interested_library,None) 


    def draw_library(i_lib,i_posi):
        count = 0
        sketch_idx = -1
        for i in range(3):
            for j in range(3):
                ax[i,j].clear()
                ax[i,j].spines['right'].set_color('none')
                ax[i,j].spines['top'].set_color('none')
        if i_lib != "None":
            # #investigate individual sketchs
            while count < 9 and sketch_idx < data_size-1:
                sketch_idx += 1
                sketch = sketch_data["sequences"][sketch_idx]
                t_ = group_type[sketch_idx]
                b_ = group_belong[sketch_idx]
                library_select_list = lib_select[sketch_idx].tolist()
                library_list = t_[:,1].tolist()

                if i_lib not in library_select_list:
                    continue


                colour_list = [int(b_[idx]/AbstractionLevel) if ttype == i_lib else -1 for idx,ttype in enumerate(library_list) ]
                high_entity = []
                if i_posi != None:
                   high_entity = [str(idx) for idx,c in enumerate(colour_list) if c != -1 and b_[idx]%AbstractionLevel == i_posi]
                
                high_entity2 = []
                for idx in high_entity:
                    if int(idx) >= len(sketch.entities):
                        c_idx = int(idx) - len(sketch.entities) +1
                        high_entity2.extend([string_strip(param.value) for param in sketch.constraints["c_"+str(c_idx)].parameters if type(param) == LocalReferenceParameter])
                    else:
                        high_entity2.append(idx)

                x_coordiante = math.floor(count/3)
                y_coordinate = count % 3
                ax[x_coordiante,y_coordinate].set_title(str(sketch_idx))
                ax[x_coordiante,y_coordinate].set_aspect('equal', 'box')
                render_sketch(sketch,high_light_entity = high_entity2,ax = ax[x_coordiante,y_coordinate],color_list = colour_list,  group_type = t_, group_belong = b_,rescale = True,show_axes = True)
                count += 1

            if count == 0:
                for i in range(3):
                    for j in range(3):
                        ax[i,j].clear()
                ax[0,1].set_title("No sketch use this library")
        plt.draw()


    def radio_reaction(event):
        global radio
        global lib_gl,lib_level
        if (radio.ignore(event) or event.mouseevent.button != 1 or
            event.mouseevent.inaxes != radio.ax):
            return
        
        if event.artist in radio.circles:
            radio.set_active(radio.circles.index(event.artist))
        interested_posi = radio.circles.index(event.artist)
        draw_library(lib_gl,interested_posi)


    axcolor = 'lightgoldenrodyellow'
    tax = plt.axes([0.1, 0.95, 0.20, 0.03], facecolor=axcolor, frameon=True)


    rax = plt.axes([0.3,0.9,0.55,0.1])
    # radio =  MyRadioButtons(rax ,[str(i)for i in range(6)], ncol=4)
    # radio.connect_event('pick_event', radio_reaction)      




    text_box = TextBox(tax, "library")
    text_box.on_submit(text_reaction)
    plt.show()
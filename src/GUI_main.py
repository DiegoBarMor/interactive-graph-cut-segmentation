import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from utilities import prompt_select_file
from mincut_maxflow import SOURCE, SINK, \
    GraphCutFastBFS, \
    GraphCutFastRND, \
    GraphCutAnimateBFS, \
    GraphCutAnimateRND
    
################################################################################
##### gui colors
NO_COLOR = 0, 0, 0, 0
RED = 255, 0, 0, 255
BLUE = 0, 0, 255, 255

HALF_RED = 255, 0, 0, 128
HALF_BLUE = 0, 0, 255, 128
HALF_BLACK = 0, 0, 0, 128

##### gui flags
MOVE = 0
DRAW = 1
ERASE = 2

##### default gui parameters
BRUSH_RADIUS = 10
DEFAULT_SIGMA = .1  # .01, .2, .5, 1
DEFAULT_LAMBDA = .2  # .001

# //////////////////////////////////////////////////////////////////////////////
class GUI:
    def __init__(self):
        ##### ATTRIBUTES INITIALIZATION
        self.im_arr = {}
        self.implot = {}

        self.str_sigma = str(DEFAULT_SIGMA)
        self.str_lambda = str(DEFAULT_LAMBDA)

        ##### IMAGE LOADING
        self.path_img = Path(prompt_select_file())

        if not self.path_img.name:
            print("XXX No image file selected. Exiting...")
            exit()
        try:
            self.img = Image.open(self.path_img)
        except UnidentifiedImageError:
            print(f"XXX Invalid image file '{self.path_img.name}' selected. Exiting...")
            exit()
        else:
            self.path_base = self.path_img.parent / self.path_img.stem
            self.path_seeds = Path(str(self.path_base) + "-seeds.png")


        ##### GUI INITIALIZATION
        fig = plt.figure()
        self.fleft, self.fright = fig.subfigures(ncols = 2, width_ratios = [8, 2])

        self.ax_left = self.fleft.add_subplot()
        self.ax_dict = self.fright.subplot_mosaic(
            "ab;cc;gg;.e;.f;hp;ij;kl;mn",
            gridspec_kw = dict(wspace = 0, hspace = 0),
        )

        self.fleft.subplots_adjust(bottom = 0.05, top = 0.95)
        self.fright.subplots_adjust(bottom = 0.05, top = 0.95)

        self.load_img(self.img, "main", True)

        self.blank = np.zeros((self.h, self.w, 4), dtype = float)
        for key in ("seeds", "ghost", "result", "paths"):
            self.im_arr[key] = np.zeros((self.h, self.w, 4), dtype = np.uint8)
            self.implot[key] = self.ax_left.imshow(self.im_arr[key], vmin = 0, vmax = 255)

        self.b_clear_seeds = Button(self.ax_dict['a'], "Clear\nSeeds")
        self.b_load_seeds = Button(self.ax_dict['b'], "Load\nSeeds")
        self.rb_gui_mode = RadioButtons(self.ax_dict['c'], ("OBJ (Z)", "BKG (X)", "Erase (C)", "Nothing (V)"))
        self.tb_sigma = TextBox(self.ax_dict['e'], "Sigma", initial = self.str_sigma)
        self.tb_lambda = TextBox(self.ax_dict['f'], "Lambda", initial = self.str_lambda)
        self.b_save_seeds = Button(self.ax_dict['g'], "Save\nseeds")
        self.b_save_result_obj = Button(self.ax_dict['h'], "Save\nOBJ", color = "0.2")
        self.b_save_result_bkg = Button(self.ax_dict['p'], "Save\nBKG", color = "0.2")
        self.b_cut_fast_bfs = Button(self.ax_dict['i'], "Fast\ncut BFS")
        self.b_cut_fast_rnd = Button(self.ax_dict['j'], "Fast\ncut RND")
        self.b_cut_animate_bfs = Button(self.ax_dict['k'], "Animate\ncut BFS")
        self.b_cut_animate_rnd = Button(self.ax_dict['l'], "Animate\ncut RND")
        self.b_toggle_result = Button(self.ax_dict['m'], "Toggle\nresult", color = "0.2")
        self.b_toggle_paths = Button(self.ax_dict['n'], "Toggle\npaths", color = "0.2")

        self.b_clear_seeds.on_clicked(self.clear_seeds)
        self.b_load_seeds.on_clicked(self.load_seeds)
        self.rb_gui_mode.on_clicked(self.update_guimode)
        self.tb_sigma.on_text_change(self.validate_tb_sigma)
        self.tb_lambda.on_text_change(self.validate_tb_lambda)
        self.b_save_seeds.on_clicked(self.save_seeds)
        self.b_save_result_obj.on_clicked(self.save_result_obj)
        self.b_save_result_bkg.on_clicked(self.save_result_bkg)
        self.b_cut_fast_bfs.on_clicked(self.trigger_cut_fast_bfs)
        self.b_cut_animate_bfs.on_clicked(self.trigger_cut_animate_bfs)
        self.b_cut_fast_rnd.on_clicked(self.trigger_cut_fast_rnd)
        self.b_cut_animate_rnd.on_clicked(self.trigger_cut_animate_rnd)
        self.b_toggle_result.on_clicked(self.toggle_result)
        self.b_toggle_paths.on_clicked(self.toggle_paths)

        self.b_save_result_obj.active = False
        self.b_save_result_bkg.active = False
        self.b_toggle_result.active = False
        self.b_toggle_paths.active = False


        self.brush_radius = BRUSH_RADIUS
        self.pressed = False
        self.colors = {}

        ### start in OBJ mode
        self.update_guimode("OBJ (Z)")
        self.gui_mode = DRAW
        self.firstCut = True
        self.showing_result = False
        self.showing_paths = False

        ### hook keyboard and mouse
        self.cid_press    = self.fleft.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_motion   = self.fleft.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release  = self.fleft.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_keyboard = self.fleft.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.details_init()
        plt.show()


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ I/O OPERATIONS
    def load_img(self, img : Image, arr_key : str, doBW : bool):
        if doBW: img = img.convert('L') # open the image in intensity mode (monochromatic)
        arr = np.asarray(img, dtype = np.uint8)

        self.im_arr[arr_key] = arr.copy()
        self.implot[arr_key] = self.ax_left.imshow(arr, cmap = "Greys_r" if doBW else None)

        self.w, self.h = img.size

    def save_masked_img(self, mask : np.array, path_out : str):
        arr = np.asarray(
            Image.open(self.path_img).convert("RGBA"),
            dtype = np.uint8
        ).copy()
        arr[~mask] *= 0
        
        p = Path(path_out)
        Image.fromarray(arr).save(p)
        print(f">>> Output saved as {p.name}")


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ GUI UTILITIES
    def get_mouse_coords(self, event):
        if event.xdata is None or event.ydata is None or event.inaxes != self.ax_left:
            self.clear_ghost(); return

        xc = abs(round(event.xdata))
        x0 = max(0, xc - self.brush_radius)
        x1 = min(self.w, xc + self.brush_radius)

        yc = abs(round(event.ydata))
        y0 = max(0, yc - self.brush_radius)
        y1 = min(self.h, yc + self.brush_radius)

        return xc, yc, x0, x1, y0, y1

    def draw_pixels(self, key, points = None):
        if self.gui_mode == MOVE: return
        if points is None: return

        xc, yc, x0, x1, y0, y1 = points
        w,h,_ = self.im_arr[key][y0:y1, x0:x1].shape

        ids = np.indices((h, w), dtype = float).T
        ids[:,:,0] += x0 - xc
        ids[:,:,1] += y0 - yc

        mask = np.sqrt(np.power(ids[:,:,0], 2) + np.power(ids[:,:,1], 2)) <= self.brush_radius
        self.im_arr[key][y0:y1, x0:x1][mask] = self.colors[key]

        self.update_canvas(key)

    def validate_textbox(self, tb_label, val):
        k_tb = f"tb_{tb_label}"
        k_str = f"str_{tb_label}"
        try:
            val = float(val)
        except ValueError:
            self.__dict__[k_tb].set_val( self.__dict__[k_str] )
        else:
            self.__dict__[k_str] = val

    def activate_button(self, b_label):
        button = self.__dict__[f"b_{b_label}"]
        button.color = .85, .85, .85
        button.active = True
        self.fright.canvas.draw()


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CANVAS UPDATES
    def clear_canvas(self, key):
        self.implot[key].set_data(self.blank)
        self.fleft.canvas.draw_idle()
    
    def update_canvas(self, key):
        self.implot[key].set_data(self.im_arr[key])
        self.fleft.canvas.draw_idle()

    def update_canvas_cut(self):
        self.im_arr["result"].fill(0)
        self.im_arr["result"] = self.graph_cut.output_array()
        
        path = self.graph_cut.get_array_path_ST()
        self.im_arr["result"] += path
        self.im_arr["paths"] += path

        self.update_canvas("result")

    def clear_ghost(self):
        self.im_arr["ghost"].fill(0)
        self.clear_canvas("ghost")

    def clear_seeds(self, event): # <fright> WIDGET CALLBACK
        self.im_arr["seeds"].fill(0)
        self.clear_canvas("seeds")

    def draw_ghost(self, points):
        self.im_arr["ghost"].fill(0)
        self.draw_pixels("ghost", points)

    def draw_seeds(self, points):
        self.draw_pixels("seeds", points)


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ KEYBOARD EVENT CALLBACKS
    def on_key_press(self, event):
        k = event.key.lower()
        if   k == '+': self.brush_radius += 1
        elif k == '-': self.brush_radius -= 1
        elif k == 'z': self.rb_gui_mode.set_active(0)
        elif k == 'x': self.rb_gui_mode.set_active(1)
        elif k == 'c': self.rb_gui_mode.set_active(2)
        elif k == 'v': self.rb_gui_mode.set_active(3)
        self.draw_ghost(self.get_mouse_coords(event))


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <fleft> MOUSE EVENT CALLBACKS
    def on_click(self, event):
        self.pressed = True
        self.draw_seeds(self.get_mouse_coords(event))

    def on_motion(self, event):
        points = self.get_mouse_coords(event)

        if self.pressed:
            self.draw_seeds(points)
            if self.gui_mode == ERASE:
                self.draw_ghost(points)
        else:
            self.draw_ghost(points)

    def on_release(self, event):
        self.pressed = False


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <fright> WIDGET CALLBACKS
    def load_seeds(self, event):
        if not self.path_seeds.exists():
            print("XXX No seeds are available for this image!")
            return
        self.clear_seeds(event)
        self.load_img(Image.open(self.path_seeds), "seeds", False)

    def update_guimode(self, val):
        if val == "OBJ (Z)":
            self.gui_mode = DRAW
            self.colors["seeds"] = RED
            self.colors["ghost"] = HALF_RED
            return

        if val == "BKG (X)":
            self.gui_mode = DRAW
            self.colors["seeds"] = BLUE
            self.colors["ghost"] = HALF_BLUE
            return

        if val == "Erase (C)":
            self.gui_mode = ERASE
            self.colors["seeds"] = NO_COLOR
            self.colors["ghost"] = HALF_BLACK
            return
    
        if val == "Nothing (V)":
            self.gui_mode = MOVE
            self.clear_canvas("ghost")

    def validate_tb_sigma(self, val):
        self.validate_textbox("sigma", val)
    
    def validate_tb_lambda(self, val):
        self.validate_textbox("lambda", val)

    def save_seeds(self, event): 
        Image.fromarray(self.im_arr["seeds"], mode = "RGBA").save(self.path_seeds)
        print(f">>> Seeds saved as {self.path_seeds.name}")

    def save_result_obj(self, event):
        self.save_masked_img(
            self.graph_cut.TREE == SOURCE, 
            f"{self.path_base}-OBJ-{'RND' if self.mode_random else 'BFS'}" \
                f"-S{self.str_sigma}-L{self.str_lambda}-cycles_{self.graph_cut.cut_cycle}.png"
        )

    def save_result_bkg(self, event):
        self.save_masked_img(
            self.graph_cut.TREE == SINK, 
            f"{self.path_base}-BKG-{'RND' if self.mode_random else 'BFS'}" \
                f"-S{self.str_sigma}-L{self.str_lambda}-cycles_{self.graph_cut.cut_cycle}.png"
        )

    def trigger_cut_fast_bfs(self, event): 
        self.mode_animate = False
        self.mode_random = False
        self.start_cut(GraphCutFastBFS)

    def trigger_cut_fast_rnd(self, event): 
        self.mode_animate = False
        self.mode_random = True
        self.start_cut(GraphCutFastRND)    

    def trigger_cut_animate_bfs(self, event): 
        self.mode_animate = True
        self.mode_random = False
        self.start_cut(GraphCutAnimateBFS)

    def trigger_cut_animate_rnd(self, event): 
        self.mode_animate = True
        self.mode_random = True
        self.start_cut(GraphCutAnimateRND)

    def toggle_result(self, event):
        self.showing_result = not self.showing_result
        f = self.update_canvas if self.showing_result else self.clear_canvas
        f("result")

    def toggle_paths(self, event):
        self.showing_paths = not self.showing_paths
        f = self.update_canvas if self.showing_paths else self.clear_canvas
        f("paths")


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ GRAPH CUT INTERFACE
    def start_cut(self, cls_cut):
        if not np.any(self.im_arr["seeds"]):
            print("XXX Can't cut if no seeds are provided!"); return

        if self.firstCut:
            self.firstCut = False
            self.graph_cut = cls_cut(
                self.im_arr["main"], self.im_arr["seeds"], 
                float(self.str_sigma), float(self.str_lambda)
            )
            self.details_add_data()

        if self.mode_animate:
            self.graph_cut.start_cut()
            self.timer = self.fleft.canvas.new_timer(interval = 0, callbacks = [(self.cut_and_update, [], {})])
            self.timer.start()

        else:
            self.graph_cut.start_cut()
            self.update_canvas_cut()
            self.end_cut()

    def cut_and_update(self):
        must_continue = self.graph_cut.continue_cut()
        self.update_canvas_cut()
        if not must_continue: self.end_cut()
        return must_continue
    

    def end_cut(self):
        self.showing_result = True
        self.details_add_data()
        self.activate_button("save_result_obj")
        self.activate_button("save_result_bkg")
        self.activate_button("toggle_result")
        if self.mode_animate:
            self.activate_button("toggle_paths")


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def details_init(self): pass
    def details_add_data(self): pass
    

# //////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    gui = GUI()

################################################################################

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from parameters import CMAP_DETAILS
from GUI_main import GUI

################################################################################
class GUIDetails(GUI):
    def details_init(self):
        self.details_data = []

        self.fig_details = plt.figure(layout = "constrained")
        self.ax_dict_details = self.fig_details.subplot_mosaic("xaabbee;xccddff")
        self.implot_details = {
            "Bp_l" : self.ax_dict_details['a'].imshow(self.im_arr["main"] / 255, vmin = 0, vmax = 1, cmap = CMAP_DETAILS),
            "Bp_r" : self.ax_dict_details['b'].imshow(self.im_arr["main"] / 255, vmin = 0, vmax = 1, cmap = CMAP_DETAILS),
            "Bp_t" : self.ax_dict_details['c'].imshow(self.im_arr["main"] / 255, vmin = 0, vmax = 1, cmap = CMAP_DETAILS),
            "Bp_b" : self.ax_dict_details['d'].imshow(self.im_arr["main"] / 255, vmin = 0, vmax = 1, cmap = CMAP_DETAILS),
            "Rp_obj" : self.ax_dict_details['e'].imshow(self.im_arr["main"] / 128, vmin = 0, vmax = 2, cmap = CMAP_DETAILS),
            "Rp_bkg" : self.ax_dict_details['f'].imshow(self.im_arr["main"] / 128, vmin = 0, vmax = 2, cmap = CMAP_DETAILS),
        }
        self.ax_dict_details['a'].set_title("Bp left")
        self.ax_dict_details['b'].set_title("Bp right")
        self.ax_dict_details['c'].set_title("Bp top")
        self.ax_dict_details['d'].set_title("Bp bottom")
        self.ax_dict_details['e'].set_title("Rp obj")
        self.ax_dict_details['f'].set_title("Rp bck")

        self.sl_details = Slider(
            ax = self.ax_dict_details['x'],
            label = "Cut index", orientation = "vertical",
            valstep = 1, valinit = 0,
            valmin = 0, valmax = 1
        )
        self.sl_details.on_changed(self.details_update_canvas)
        self.sl_details.active = False

    def details_update_canvas(self, data_index):
        for k,data in self.details_data[data_index].items():
            self.implot_details[k].set_data(data)
        self.fig_details.canvas.draw_idle()

    def details_add_data(self):
        self.sl_details.active = True

        mat_Bp_left, mat_Bp_right, mat_Bp_top, mat_Bp_bottom = self.graph_cut.get_arrays_bp()
        mat_Rp_obj, mat_Rp_bkg = self.graph_cut.get_arrays_rp()
        self.details_data.append(dict(
            Bp_l = mat_Bp_left,
            Bp_r = mat_Bp_right,
            Bp_t = mat_Bp_top,
            Bp_b = mat_Bp_bottom,
            Rp_obj = mat_Rp_obj,
            Rp_bkg = mat_Rp_bkg,
        ))

        idx = len(self.details_data) - 1
        if idx > 0:
            self.sl_details.valmax = idx
            self.ax_dict_details['x'].set_ylim(0, idx)
        self.details_update_canvas(idx)
        self.sl_details.set_val(idx)


# //////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    gui = GUIDetails()

################################################################################

"""Creates a GUI containing a dropdown and the terra map. The user selects the faction from the
drop down & the map displays tiles according to distance from that faction's home times.
"""

import tkinter
import terravisualisation as tmvis
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Implement the default Matplotlib key bindings.
from matplotlib.figure import Figure

root = tkinter.Tk()
root.wm_title("Faction map")

# add the photo
root.iconbitmap('D:/PycharmProjects/TerraBot/images/tmbotico.ico')

# make the dropdown
factions = {'Witches', 'Auren', 'Giants', 'Chaos Magicians', 'Darklings', 'Alchemists',
            'Swarmlings', 'Mermaids', 'Fakirs', 'Nomads', 'Engineers', 'Dwarves', 'Halflings', 'Cultists'}

tkvar = tkinter.StringVar(root)
tkvar.set('Witches')

popupMenu = tkinter.OptionMenu(root, tkvar, *factions)
popupMenu.pack(side=tkinter.TOP)

# make the figure
fig = Figure(figsize=(5, 4), dpi=100)
a = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.


def createtheplot(faction):
    x, y = tmvis.display_map(faction, plot=False)

    a.hexbin(x, y, gridsize=(19, 9), cmap='magma')
    a.axis('off')

    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


createtheplot(tkvar.get())


# on change dropdown value
def change_dropdown(*args):
    createtheplot(tkvar.get())


# link function to change dropdown
tkvar.trace('w', change_dropdown)


# make the quit button
def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager


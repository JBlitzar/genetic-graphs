import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from PIL import Image

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 4, figure=fig)
fig.show()

entities = []
axes = []
i = 0
def reInit():
    global fig, gs, i
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig)
    fig.show()


    i = 0

def onLoop(img,value):
    global i, axes, fig, gs
    

    img = Image.open(img)
    if i >= len(axes):
        nrows = len(axes) // 4 + 1  
        fig.clf()  
        gs = GridSpec(nrows, 4, figure=fig) 
        axes = [fig.add_subplot(gs[j // 4, j % 4]) for j in range(nrows * 4)] 
    

    ax = axes[i]
    ax.clear()
    ax.imshow(img)
    ax.set_title(f"Value: {value}")
    ax.axis('off') 
    
    fig.canvas.draw()  
    fig.canvas.flush_events()

    i+= 1


    plt.show(block=False)
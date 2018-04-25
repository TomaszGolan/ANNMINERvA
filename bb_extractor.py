"""The script creates a new HDF5 file with data cropped to the bounding box."""

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pylab
import numpy as np
import click


def load_dataset(hdf5_in):
    """Load energy+times and pid data."""
    f = h5py.File(hdf5_in, 'r')
    try:
        return f['img_data']['hitimes-x'], f["pid_data"]["pid-x"]
    except KeyError:
        return f['hitimes-x'], f['pid-x']


def save_dataset(hdf5_out, data):
    """Save energy and pid data to a new file."""
    f = h5py.File(hdf5_out, 'w')

    for key in data:
        f.create_dataset(key, data[key].shape, dtype=data[key].dtype,
                         compression='gzip')[...] = data[key]

    f.close()


def find_bbox(hitimes, width, height):
    """Find the bounding box position.
    
    The score is just a sum of all pixels in the box weighted so "earlier" hits
    weight more.
    """
    best_sum = 0  # best score
    best_x = 0    # best box x position
    best_y = 0    # best box y position

    data = hitimes[0] / np.max(hitimes[0])  # data = normalized energy
    data[data < 0.1] = 0                    # remove "noise" (low energy hits)
    
    time = hitimes[1] + np.max(hitimes[1]) + 1  # make time positive
    time = time / np.max(time)                  # normalize
    time = np.square(time)                      # I got better result with t^2

    data = np.divide(data, time)  # normalized energy weighted by time
    
    # consider all possible position of a box width x height
    for x in range(0, data.shape[0] - width):
        for y in range(0, data.shape[1] - height):
            box = data[x:x+width, y:y+height]
            score = np.sum(box)

            # update box if better one is found
            if score > best_sum:
                best_sum = score
                best_x = x
                best_y = y
    
    return best_x, best_y

##### MAIN #####

@click.group()
def main():
    pass


@main.command()
@click.option('--hdf5_in', help='Original Jon-like HDF5 file', required=True)
@click.option('--pdf_out', help='Output pdf file', default="bb_test.pdf")
@click.option('--width', help='Width of bounding box', default=50)
@click.option('--height', help='Height of bounding box', default=50)
@click.option('--n_events', help='Number of event to plot', default=10)   
def plot_bb(hdf5_in, pdf_out, width, height, n_events):
    """Generate pdf with bounding box plotted on the img."""
    hitimes, pids = load_dataset(hdf5_in)

    # labels
    pid_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
    pid_labels = ['nth', 'EM', 'mu', 'pi+', 'pi-', 'n', 'p', 'oth']

    with PdfPages(pdf_out) as pdf:
        for _ in range(n_events):
            # grab random event
            i = np.random.randint(0, hitimes.shape[0])
            
            fig,ax = plt.subplots(1)

            cmap = 'tab10'
            cbt = 'pid'

            # plot pid image
            im = ax.imshow(pids[i][0], cmap=pylab.get_cmap(cmap), vmin=0, vmax=7)

            # just plot settings
            cbar = pylab.colorbar(im, fraction=0.04, ticks=pid_ticks)
            cbar.ax.set_yticklabels(pid_labels)
            cbar.set_label("pid", size=9)
            cbar.ax.tick_params(labelsize=6)
            pylab.title("event: " + str(i), fontsize=12)

            # find bounding box
            x, y = find_bbox(hitimes[i], width, height)

            # plot bounding box
            rect = patches.Rectangle((y,x), height, width, linewidth=1,
                                     edgecolor='1', facecolor='none')
            ax.add_patch(rect)

            pdf.savefig()


@main.command()
@click.option('--hdf5_in', help='Original Jon-like HDF5 file', required=True)
@click.option('--hdf5_out', help='Cropped HDF5 file', required=True)
@click.option('--pdf_out', help='Output pdf file', default="crop_test.pdf")
@click.option('--n_events', help='Number of event to plot', default=10)   
def compare(hdf5_in, hdf5_out, pdf_out, n_events):
    """Plots original and cropped img side by side."""
    _, original = load_dataset(hdf5_in)   # original dataset
    _, cropped = load_dataset(hdf5_out)  # cropped dataset

    # labels
    pid_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
    pid_labels = ['nth', 'EM', 'mu', 'pi+', 'pi-', 'n', 'p', 'oth']

    with PdfPages(pdf_out) as pdf:
        for _ in range(n_events):
            # grab random event
            i = np.random.randint(0, cropped.shape[0])
            
            fig, axes = plt.subplots(1, 2)

            cmap = 'tab10'
            cbt = 'pid'

            fig.suptitle("Event: " + str(i))

            # plot pid image
            im = axes[0].imshow(original[i][0], cmap=pylab.get_cmap(cmap),
                                vmin=0, vmax=7)
            # plot pid image
            im = axes[1].imshow(cropped[i], cmap=pylab.get_cmap(cmap),
                                vmin=0, vmax=7)

            # just plot settings
            cbar = pylab.colorbar(im, fraction=0.04, ticks=pid_ticks)
            cbar.ax.set_yticklabels(pid_labels)
            cbar.set_label("pid", size=9)
            cbar.ax.tick_params(labelsize=6)

            pdf.savefig()


@main.command()
@click.option('--hdf5_in', help='Original Jon-like HDF5 file', required=True)
@click.option('--hdf5_out', help='Target HDF5 file', required=True)
@click.option('--width', help='Width of bounding box', default=50)
@click.option('--height', help='Height of bounding box', default=50)
@click.option('--n_events', help='Number of event to plot', default=10)   
def create_dataset(hdf5_in, hdf5_out, width, height, n_events=0):
    """Create hdf5_out with data cropped to bounding box"""
    hitimes, pids = load_dataset(hdf5_in)
    
    data = {}  # placeholder for cropped data
    data['hitimes-x'] = np.empty((n_events or hitimes.shape[0], width, height))
    data['pid-x'] = np.empty((n_events or hitimes.shape[0], width, height))

    for i in range(n_events or hitimes.shape[0]):
        # get position of the bounding box
        x, y = find_bbox(hitimes[i], width, height)
        # crop data
        energy = hitimes[i][0][x:x+width, y:y+height]
        pid = pids[i][0][x:x+width, y:y+height]
        # save into dict
        data['hitimes-x'][i] = energy
        data['pid-x'][i] = pid

    save_dataset(hdf5_out, data)


if __name__ == "__main__":
    main()
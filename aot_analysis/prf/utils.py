from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt

from aot_analysis import io_utils


def test_alignment(timepoints):
    # check that all events are associated with exactly one trial
    assert timepoints.trial_nr_min.equals(
        timepoints.trial_nr_max), "Some timepoints are associated with more than one trial"

    # check that all timepoints are associated with up to one aperture
    aperture_indices = ~(timepoints.seq_index_min.isna() |
                         timepoints.seq_index_max.isna())
    aperture_mismatch = timepoints.seq_index_min != timepoints.seq_index_max
    warn_indices = aperture_indices & aperture_mismatch
    if warn_indices.sum() > 0:
        print('Warning: Some timepoints are associated with more than one aperture')
        print(timepoints.loc[warn_indices])


def create_gif(design_matrix, tr, label, subject, run):
    num_frames = design_matrix.shape[0]
    print(design_matrix.shape)

    images = []
    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(design_matrix[i, :, :], origin='lower')
        ax.set_title(f'Design Matrix for Run {run}')
        ax.axis('off')

        # Save the frame as an image in memory
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        plt.close(fig)  # Close the figure to avoid memory issues

    # Save the sequence as a GIF
    gif_filename = DIR_OUTPUT / \
        (f"sub-{str(subject).zfill(2)}"
         + f"_run-{str(run).zfill(2)}_design_matrix_{label}.gif")
    imageio.mimsave(gif_filename, images, fps=1/tr)
    print(f"Saved GIF: {gif_filename}")


core_settings = io_utils.load_config()
params = core_settings['prf']['design_matrix']
DIR_BASE = Path(core_settings['paths']['prf_experiment']['base'])
DIR_INPUT = DIR_BASE / core_settings['paths']['prf_experiment']['input']
DIR_OUTPUT = DIR_BASE / core_settings['paths']['prf_experiment']['design']
BLANK = params['blank']

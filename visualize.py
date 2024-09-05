import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def np_to_pil(img_np, input_name, cmap='gist_ncar'):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    plt.imshow(ar, cmap=cmap)
    plt.axis('off')  # Remove axis
    plt.tight_layout(pad=0)  # Remove padding

    # Save the plot as a PIL image
    plt.savefig(os.path.join(output_path,input_name.split('/')[-1]).replace('.npy','.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


input_filename = '/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/64X64/minmax_ttv/test/lq/0_000200.npy'
image = np.expand_dims(np.load(input_filename), axis=0)

# define output path
output_path = './visualize'
os.makedirs(output_path,exist_ok=True)
p = np_to_pil(image, input_filename, cmap='gist_ncar')

print(f"Image saved as {output_path}")

import os
import numpy as np
import h5py

out_root = "/mnt/home/liangdi/projects/pl_version/ls_eend/outputs/data/onl_allspk_version_10w_ver_204_ch_ver_88/preds"
out_h5_root = "/mnt/home/liangdi/projects/pl_version/ls_eend/outputs/data/onl_allspk_version_10w_ver_204_ch_ver_88/preds_h5"
if not os.path.isdir(out_h5_root):
    os.mkdir(out_h5_root)
bsz = 500

for dir_path, dir_names, file_names in os.walk(out_root):
    for file_name in file_names:
        pred = np.load(os.path.join(out_root, file_name))
        pred = 1 / (1 + np.exp(-pred))
        print(pred)
        name = file_name.split('.')[0]
        h5_file = h5py.File(out_h5_root + f"/{name}.h5", "w")
        h5_file.create_dataset("T_hat", data=pred)
        h5_file.close()

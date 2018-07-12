import h5py
import  numpy as np
squad_h5_path=''

with h5py.File(squad_h5_path, 'r') as f:
    f_data = f['data']

    dev_data={}
    for sub_name in ['contents', 'question_ans', 'samples_ids', 'samples_labels', 'samples_categorys','samples_logics']:
        dev_data[sub_name] = np.array(f_data["dev"][sub_name])


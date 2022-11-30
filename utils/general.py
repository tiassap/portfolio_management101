import numpy as np

def join(loader, node):
    baseline_map = {'True':'baseline', 'False':'no-baseline'}
    seq = loader.construct_sequence(node)
    seq = [baseline_map[str(x)] if (str(x) in baseline_map.keys()) else x for x in seq]
    return ''.join([str(i) for i in seq])
    

def gen_sequence(data_matrix, seq_length):
    """
    Output sequence -> 
    for seq len=2
            0 2
            1 3
            2 4
            3 5
    """
    num_elements = data_matrix.shape[0]
    for start in range(0, num_elements-seq_length):
        stop = start+seq_length
        # print(f"start : {start}, end :{stop}")
        yield data_matrix[start:stop, :].astype(np.float32)


def generate_window():
    x_vec = []
    for feat in X:
        # for each feat open, high, low create sequences
        # feat dims[ history, num_currencies]
        feat_vec = [x for x in gen_sequence(feat, 50)]
        # feat_vec --> [ batch, window_size, num_currencies]
        x_vec.append(feat_vec)
    x_vec = np.stack(x_vec, axis=0).transpose(1, 0, 3, 2)
    x_vec.shape  # [batch_size, feature_number, num_currencies, window_size]


import numpy as np
import tensorflow as tf
from ml_loaddata_rnn import load_radscheme_rnn
import xarray as xr
import random

def load_data(file_name):

    x, _, rsd0, _, rsd, rsu, pres = \
        load_radscheme_rnn(file_name,  scale_p_h2o_o3 = True, return_p=True, return_coldry=False)
        
    xmax = np.array([3.20104980e+02, 1.15657101e+01, 4.46762830e-01, 5.68951890e-02,
        6.59545418e-04, 3.38450207e-07, 5.08802714e-06, 2.13372910e+02,
        1.96923096e+02, 1.00000000e+00, 1.00],dtype=np.float32)
    
    x = x / xmax

    nlay = x.shape[-2]

    y = np.concatenate((np.expand_dims(rsd,2), np.expand_dims(rsu,2)),axis=2)

    y = y / np.reshape(rsd0, (-1, 1, 1))

    rsd0_big = rsd0.reshape(-1,1).repeat(nlay+1,axis=1)

    # only one scalar input (albedo)
    x_m = x[:,:,0:-1]
    n_f = x_m.shape[-1]
    x_aux1 = x[:,0,-1:]   

    dp = pres[:,1:] - pres[:,0:-1] 

    dt = xr.open_dataset(file_name)
    tmp_selection = dt.variables['is_valid_zenith_angle'].data.astype(int)
    tmp_selection = np.reshape(tmp_selection, (x_m.shape[0]))
    selection = tmp_selection.astype(bool)
    dt.close()

    return x_m[selection], x_aux1[selection], y[selection], dp[selection], rsd0_big[selection]


    
class non_sequential_access(Exception):
    "Raised when InputSequence is not sequentially accessed"
    pass
    


class InputSequence(tf.keras.utils.Sequence):

    def __reshuffle(self):
        self.e_shuffle = []   # Shuffle of examples within each month
        self.m_shuffle = np.array(list(range(12))) # shuffle of months
        random.shuffle(self.m_shuffle)
        acc = 0
        for i, m in enumerate(self.m_shuffle):
            c = self.n_data[m]
            acc += c  // self.batch_size
            self.n_batch_accumulated[i] = acc
            a = np.array(list(range(c)))
            random.shuffle(a)
            self.e_shuffle.append(a)

    def __shuffle(self, x_m, x_aux1, y, dp, rsd0_big):
        e = self.e_shuffle[self.i_file]
        self.x_m = tf.convert_to_tensor(x_m[e,:,:])
        self.x_aux1 = tf.convert_to_tensor(x_aux1[e,:])
        self.y = tf.convert_to_tensor(y[e,:,:])
        self.dp = tf.convert_to_tensor(dp[e,:])
        self.rsd0_big = tf.convert_to_tensor(rsd0_big[e,:])

    def __free_memory(self):
        del self.x_m
        del self.x_aux1
        del self.y 
        del self.dp
        del self.rsd0_big 

    def __init__(self, file_names, batch_size):
        self.file_names = file_names
        self.batch_size = batch_size
        self.n_data = []
        self.n_batch_accumulated = []
        acc = 0
        for f in file_names:
            dt = xr.open_dataset(f)
            #c = int(dt['mu0'].shape[0]* dt['mu0'].shape[1])
            c = int(np.sum(dt['is_valid_zenith_angle'].data))
            dt.close()
            self.n_data.append(c)
            acc += c // self.batch_size
            self.n_batch_accumulated.append(acc)

        self.i_file = 0
        print(f"Total number of examples = {np.sum(np.array(self.n_data))}")
        print(f"Number of valid examples per epoch = {acc * self.batch_size}")
        print(f"Number of valid batches = {acc}", flush=True)

    def __len__(self):
        return self.n_batch_accumulated[-1]

    def __getitem__(self, idx):
        # idx is a batch index

        # Assumes data is accessed sequentially
        # Verify that it is

        if idx > self.n_batch_accumulated[self.i_file]:
            print (f"idx = {idx}, max-idx = {self.n_batch_accumulated[self.i_file]}")
            raise non_sequential_access
        elif idx < 0:
            print (f"idx = {idx}")
            raise non_sequential_access
        else:
            if self.i_file > 0:
                if idx < self.n_batch_accumulated[self.i_file - 1]:
                    print (f"self.i_file = {self.i_file}")
                    print (f"idx = {idx}, min-idx = {self.n_batch_accumulated[self.i_file - 1]}")
                    raise non_sequential_access
                
        if idx == 0:
            if hasattr(self, 'x_m'):
                self.__free_memory()
                del self.e_shuffle
                del self.m_shuffle
            self.__reshuffle()
            self.i_file = 0
            x_m, x_aux1, y, dp, rsd0_big = load_data(self.file_names[self.m_shuffle[self.i_file]])
            self.__shuffle(x_m, x_aux1, y, dp, rsd0_big)  

        elif idx == self.n_batch_accumulated[self.i_file]:
            self.i_file = self.i_file + 1
            self.__free_memory()
            x_m, x_aux1, y, dp, rsd0_big = load_data(self.file_names[self.m_shuffle[self.i_file]])
            self.__shuffle(x_m, x_aux1, y, dp, rsd0_big)  

        if self.i_file == 0:
            i2 = idx
        else:
            i2 = idx - self.n_batch_accumulated[self.i_file - 1]

        i = i2 * self.batch_size
        j = (i2 + 1) * self.batch_size

        return ([self.x_m[i:j], self.x_aux1[i:j], self.y[i:j], self.dp[i:j], self.rsd0_big[i:j]], self.y[i:j])
        
    def on_epoch_end(self): 
        self.i_file = 0


def create_data_generator(file_names, scale_inputs=True):

    nlay=60
    nx_main = 10 # 5-gases + temp + pressure + mu + lwp + iwp

    signature = (tf.TensorSpec(shape=(nlay,nx_main),dtype=tf.float32),
                tf.TensorSpec(shape=(1,),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay+1,2),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay+1),dtype=tf.float32))
    
    def generator(file_names, scale_inputs=True):
        for f in file_names:
            print(f"Opening {f}")
            
            x_m, x_aux1, y, dp, rsd0_big = load_data(f)

            for i,_ in enumerate(x_m):
                yield (x_m[i], x_aux1[i], y[i], dp[i], rsd0_big[i])
    if True:
        return tf.data.Dataset.from_generator(generator,
                                          args=[file_names,True],
                                          output_signature=signature)
    else:
        return generator(file_names,scale_inputs)
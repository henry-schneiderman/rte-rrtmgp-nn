import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

data_dir       = "/data-T1/hws/tmp/"

#file_name_2009 = data_dir + '../CAMS/processed_data/testing/2009/internal_output.sc_v1_tau_4_460.nc'
t = "505" #"480"
year = "2009"
#version = "v5_e19"
version = "v5_28"
file_name_2009 = data_dir + f'../CAMS/processed_data/testing/{year}/internal_output.sc_{version}_{t}.{year}.nc'
dt_2009 = xr.open_dataset(file_name_2009)
is_direct = False
is_scattering = False

s_direct = dt_2009['s_direct'].data
s_diffuse = dt_2009['s_diffuse'].data
t_direct = dt_2009['t_direct'].data
t_diffuse = dt_2009['t_diffuse'].data
o3 = dt_2009['o3'].data
h2o = dt_2009['h2o'].data
mu_diffuse = dt_2009['mu_diffuse'].data
mu_direct = dt_2009['mu_direct'].data
lwp = dt_2009['lwp'].data
iwp = dt_2009['iwp'].data
r_toa = dt_2009['r_toa'].data
r_surface = dt_2009['r_surface'].data

wp = lwp + iwp

print(f"mean r_direct = {np.mean(s_direct)}")
print(f"mean r_diffuse = {np.mean(s_diffuse)}")
print(f"mean mu_diffuse = {np.mean(mu_diffuse)}")
print(f"std mu_diffuse = {np.std(mu_diffuse)}")

shape = s_direct.shape
eps = 0.000001

mu_direct = np.reshape(mu_direct,(len(mu_direct), 1))
mu_direct_squeezed = np.reshape(mu_direct,(-1,))
mu_direct_indices = np.argsort(mu_direct_squeezed)
mu_direct_ranked = mu_direct_squeezed[mu_direct_indices]
l = len(mu_direct_squeezed)
print (f"min(mu_direct) = {np.min(mu_direct)}")
print (f"max(mu_direct) = {np.max(mu_direct)}")
print(f"mu direct 0.0%: {mu_direct_ranked[0]}")
print(f"mu direct 0.1%: {mu_direct_ranked[l//1000]}")
print(f"mu direct 1%: {mu_direct_ranked[l//100]}")
print(f"mu direct 5%: {mu_direct_ranked[l // 20]}")
print(f"mu direct 10%: {mu_direct_ranked[l // 10]}")
print(f"mu direct 20%: {mu_direct_ranked[l // 5]}")
print(f"mu direct 33%: {mu_direct_ranked[l // 3]}")
print(f"mu direct 50%: {mu_direct_ranked[l // 2]}")
print(f"mu direct 75%: {mu_direct_ranked[(l * 3) // 4]}")
print(f"mu direct 100%: {mu_direct_ranked[l-1]}")



if is_direct:
    wp = wp / (mu_direct + eps)
    h2o = h2o / (mu_direct + eps)
    o3 = o3 / (mu_direct + eps)

wp = wp.reshape((shape[0] * shape[1]))
h2o = h2o.reshape((shape[0] * shape[1]))
o3 = o3.reshape((shape[0] * shape[1]))

s_direct = s_direct.reshape((shape[0] * shape[1]))
s_diffuse = s_diffuse.reshape((shape[0] * shape[1]))
t_direct = t_direct.reshape((shape[0] * shape[1]))
t_diffuse = t_diffuse.reshape((shape[0] * shape[1]))


h2o_ranked_indices = np.argsort(h2o)
#h2o_ranked_indices = h2o_ranked_indices[::-1]
h2o_rank_fraction = np.zeros(len(h2o),dtype=np.float32)
for i,r in enumerate(h2o_ranked_indices):
    h2o_rank_fraction[r] = float(i) / float(len(h2o))


o3_ranked_indices = np.argsort(o3)
#o3_ranked_indices = o3_ranked_indices[::-1]
o3_rank_fraction = np.zeros(len(o3),dtype=np.float32)
for i,r in enumerate(o3_ranked_indices):
    o3_rank_fraction[r] = float(i) / float(len(o3))


wp_ranked_indices = np.argsort(wp)
#wp_ranked_indices = wp_ranked_indices[::-1]
wp_rank_fraction = np.zeros(len(wp),dtype=np.float32)
for i,r in enumerate(wp_ranked_indices):
    wp_rank_fraction[r] = float(i) / float(len(wp))

t_direct_ranked_indices = np.argsort(t_direct)
#t_direct_ranked_indices = t_direct_ranked_indices[::-1]

rank_index = len(h2o) - 40000
h2o_value_fraction = h2o / h2o[h2o_ranked_indices[rank_index]]
o3_value_fraction = o3 / o3[o3_ranked_indices[rank_index]]
wp_value_fraction = wp / wp[wp_ranked_indices[rank_index]]

plt.xlim(0, 1)
plt.ylim(0, 1)
#plt.scatter(r_toa_ordered[:100000], wp_toa_ordered[:100000], c='#1f77b4') #c='#ff7f0e' c='#1f77b4'

#plt.scatter(r_toa_ordered[:100000], d * 50.0 * o3_toa_ordered[:100000], c='#1f77b4') #c='#ff7f0e' c='#1f77b4'
sample_size = len(t_direct) #12000
if is_direct:
    if is_scattering:
        plt.scatter(wp_rank_fraction[:sample_size], s_direct[:sample_size], # 
            c='#ff7f0e', marker=".", s=1.0)
        plt.xlabel("Cloud Content (ranked)")
        plt.ylabel("Fraction Scattered")
        plt.title(f"Direct Radiation")# sigma={sigma}")# train_r={train_radius} test_r={test_radius}")
    else:
        plt.scatter(0.5 * (h2o_rank_fraction[:sample_size] + wp_rank_fraction[:sample_size]), t_direct[:sample_size], # 
            c='#ff7f0e', marker=".", s=1.0)
        plt.title ("Direct Radiation")
        plt.ylabel("Fraction Transmitted")
        plt.xlabel("Cloud and H2O Content (ranked)")


else:
    if is_scattering:
        plt.scatter(wp_rank_fraction[:sample_size], s_diffuse[:sample_size], # 
            c='#ff7f0e', marker=".", s=1.0)
        plt.xlabel("Cloud Content (ranked)")
        plt.ylabel("Fraction Scattered")
        plt.title(f"Diffuse Radiation")
    else:
        plt.scatter(0.5 * (h2o_rank_fraction[:sample_size] + wp_rank_fraction[:sample_size]), t_diffuse[:sample_size], 
                # + o3_rank_fraction[:sample_size],
            c='#ff7f0e', marker=".", s=1.0) #c='#ff7f0e' c='#1f77b4'
        #plt.title(f"s_diffuse vs. wp (rank fraction)")# sigma={sigma}")# train_r={train_radius} test_r={test_radius}")
        plt.title ("Diffuse Radiation")
        #plt.ylabel("Fraction Scattered")
        plt.ylabel("Fraction Transmitted")
        plt.xlabel("Cloud and H2O Content (ranked)")

plt.show()
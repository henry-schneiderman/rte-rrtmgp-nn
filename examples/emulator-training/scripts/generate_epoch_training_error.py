from netCDF4 import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cartopy.crs as ccrs



def generate_epoch_validation_error(file_name, is_training,error_name="Loss", best_loss = 9999999.0):
    epochs = []
    errors = []
    number_of_elapsed_epochs = 0
    l = len(error_name)
    with open(file_name, "r") as f:
        for line in f:
            if not is_training and line.find("poch",0,6) > 0:
                epochs.append(line[5:].strip())
            if is_training and line.find("ed Model: epoch",3) > 0:
                epochs.append(line[21:].strip())

            #if line.find("Loss",1,6) > 0:
            if line.find(error_name,1,l+2) > 0:
                epoch = np.int32(epochs[-1])
                if epoch in [285,360,515]:
                    best_loss = 999999.0
                errors.append(line[l+2:].strip())
                loss = np.float32(errors[-1])
                if loss < best_loss:
                    if False:
                        print(f"New Best Loss. Number of Elapsed = {number_of_elapsed_epochs + 1}")
                        print(f" epoch = {epochs[-1]}, loss = {loss}")
                    best_loss = loss
                    number_of_elapsed_epochs = 0
                else:
                    number_of_elapsed_epochs = number_of_elapsed_epochs + 1
            if line.find("best",1,10) > 0:
                print(line + " epoch = " + epochs[-1])

            


    if len(epochs) == len(errors) + 1 or len(epochs) == len(errors):
        pass
        #for i,junk in enumerate(errors):
        #    print(f"{epochs[i]} {errors[i]}")
    else:
        print (f'Error: number of epochs = {len(epochs)}')
        print (f'Error: number of errors = {len(errors)}')

    return epochs, errors, best_loss


def generate_epoch_error_ukkonen(file_name, is_best_in_25):
    epochs = []
    val_loss = []
    best_loss = 9999999.0
    train_loss = []
    number_of_elapsed_epochs = 0
    last_best_n = 0
    n = 0
    largest_elapsed = 0
    if is_best_in_25:
        sample_rate = 25
    else:
        sample_rate = 1
    with open(file_name, "r") as f:
        for line in f:
            #i = line.find("- loss: ",55,70)
            i = line.find("- val_loss: ")
            if i > 0:
                n = n + 1
                epochs.append(n)

                #tmp = line[i+8:i+18]
                tmp = line[i+11:i+24]
                tmp = tmp.lstrip()
                tmp_strings = tmp.split(' ')
                loss = np.float32(tmp_strings[0])
                val_loss.append(loss)
                if loss < best_loss and n % sample_rate == 0:


                    print(f"New Best Loss. Number of Elapsed = {number_of_elapsed_epochs + 1} or {n - last_best_n}")
                    print(f"loss = {loss} epoch = {epochs[-1]} or {n}")
                    if number_of_elapsed_epochs > largest_elapsed:
                        print(f"New max elapsed = {number_of_elapsed_epochs}")
                        largest_elapsed = number_of_elapsed_epochs
                    best_loss = loss
                    last_best_n = n
                    number_of_elapsed_epochs = 0
                else:
                    number_of_elapsed_epochs = number_of_elapsed_epochs + 1

                j = line.find("step - loss: ")
                if j < 0:
                    print("Could not find training error")
                tmp = line[j+12:j+26]
                tmp = tmp.lstrip()
                tmp_strings = tmp.split(' ')
                loss = np.float32(tmp_strings[0])
                train_loss.append(loss)
                
                
            #if line.find("best",1,10) > 0:
            #    print(line + " epoch = " + epochs[-1])

            
    if False:

        if len(epochs) == len(errors) + 1 or len(epochs) == len(errors):
            pass
            #for i,junk in enumerate(errors):
            #    print(f"{epochs[i]} {errors[i]}")
        else:
            print (f'Error: number of epochs = {len(epochs)}')
            print (f'Error: number of errors = {len(errors)}')

    return epochs, val_loss, train_loss


def get_data(file_names, is_training, error_name="Loss"):
    epochs = []
    errors = []
    best_loss = 999999.0
    for file_name in file_names:
        epoch_list, error_list, best_loss = generate_epoch_validation_error(file_name, is_training, error_name, best_loss)

        if len(epoch_list) == len(error_list) + 1:
            epoch_list = epoch_list[:-1]

        epochs.extend(epoch_list)
        errors.extend(error_list)

    epochs = np.array(epochs, dtype=np.int32)
    errors = np.array(errors, dtype=np.float32)
    return epochs, errors

def get_layer_data(file_name, is_rmse=True):
    is_data_started = False
    indices = []
    errors = []
    last_index = 0
    with open(file_name, "r") as f:
        for line in f:
            if is_data_started and last_index < 59:
                components = line.split('.')
                last_index = np.int32(components[0].lstrip())
                indices.append(last_index)
                error_string = components[1] + "." + components[2]
                errors.append(np.float32(error_string.lstrip()))
            if is_rmse and line.find("rmse") > 0:
                is_data_started = True
                print(f'Starting {file_name}')
            if not is_rmse and line.find("bias") > 0:
                is_data_started = True
                print(f'Starting {file_name}')
        print(f"Last index = {last_index}")

    return indices, errors

            
def plot_geographic_results(file_name_geography="/data-T1/hws/CAMS/processed_data/testing/2009/01/lw_input-testing-2009-01.nc"):
    file_name_2009 = '/data-T1/hws/tmp/Loss_Torch.SW.v1.v1.596.2009.nc'
    file_name_2015 = '/data-T1/hws/tmp/Loss_Torch.SW.v1.v1.596.2015.nc'
    file_name_2020 = '/data-T1/hws/tmp/Loss_Torch.SW.v1.v1.596.2020.nc'
    dt_2009 = Dataset(file_name_2009, "r")
    dt_2015 = Dataset(file_name_2015, "r")
    dt_2020 = Dataset(file_name_2020, "r")
    dt_geography = Dataset(file_name_geography, "r")
    hr_2009_rmse = dt_2009.variables["heating_rate_rmse"][:].data
    up_flux_2009_rmse = dt_2009.variables["upwelling_flux_rmse"][:].data
    up_flux_2015_rmse = dt_2015.variables["upwelling_flux_rmse"][:].data
    up_flux_2020_rmse = dt_2020.variables["upwelling_flux_rmse"][:].data
    up_flux_2009_bias = dt_2009.variables["upwelling_flux_bias"][:].data
    up_flux_2015_bias = dt_2015.variables["upwelling_flux_bias"][:].data
    up_flux_2020_bias = dt_2020.variables["upwelling_flux_bias"][:].data

    down_flux_2009_rmse = dt_2009.variables["downwelling_flux_rmse"][:].data
    down_flux_2015_rmse = dt_2015.variables["downwelling_flux_rmse"][:].data
    down_flux_2020_rmse = dt_2020.variables["downwelling_flux_rmse"][:].data
    down_flux_2009_bias = dt_2009.variables["downwelling_flux_bias"][:].data
    down_flux_2015_bias = dt_2015.variables["downwelling_flux_bias"][:].data
    down_flux_2020_bias = dt_2020.variables["downwelling_flux_bias"][:].data

    heating_rate_2009_rmse = dt_2009.variables["heating_rate_rmse"][:].data
    heating_rate_2015_rmse = dt_2015.variables["heating_rate_rmse"][:].data
    heating_rate_2020_rmse = dt_2020.variables["heating_rate_rmse"][:].data
    heating_rate_2009_bias = dt_2009.variables["heating_rate_bias"][:].data
    heating_rate_2015_bias = dt_2015.variables["heating_rate_bias"][:].data
    heating_rate_2020_bias = dt_2020.variables["heating_rate_bias"][:].data

    lat = np.rad2deg(dt_geography.variables["clat"][:].data)
    lon = np.rad2deg(dt_geography.variables["clon"][:].data)

  
    proj = ccrs.EqualEarth()
    x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T
    if False:
        fig = plt.figure()
        #proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)
        #ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9],projection=proj)
        #ax.add_feature(cartopy.feature.LAND, color='tan')
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        #ax.tricontourf(lon,lat,hr_rmse)
        c1 = ax.tricontourf(x,y,hr_rmse)
        #ax.tricontourf(x,y,up_flux_rmse)
        #ax.tricontourf(x,y,down_flux_rmse)
        colorbar0 = fig.colorbar(c1)
    elif False:
        fig = plt.figure(figsize=(9.0,9.0))
        #fig.suptitle('Vertically stacked subplots')

        ax1 = plt.subplot(3, 1, 1, projection=proj, ymargin=0.0, xmargin=0.0)
        ax1.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax2 = plt.subplot(3, 1, 2, projection=proj, ymargin=0.0, xmargin=0.0)
        ax2.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax3 = plt.subplot(3, 1, 3, projection=proj, ymargin=0.0, xmargin=0.0)
        ax3.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        vmin = min([up_flux_2009_rmse.min(), up_flux_2015_rmse.min(), up_flux_2020_rmse.min()])
        vmax = max([up_flux_2009_rmse.max(), up_flux_2015_rmse.max(), up_flux_2020_rmse.max()])

        c1 = ax1.tricontourf(x,y,up_flux_2009_rmse, vmin=vmin, vmax=vmax)
        c2 = ax2.tricontourf(x,y,up_flux_2015_rmse,vmin=vmin, vmax=vmax)
        c3 = ax3.tricontourf(x,y,up_flux_2020_rmse,vmin=vmin, vmax=vmax)
        #ax1.annotate("flux up")
        #ax2.annotate("flux down")

        fig.colorbar(c3, ax=(ax1,ax2,ax3), location='bottom', shrink=0.5, fraction=0.05, pad=0.05)
    elif False:
        # Upwelling flux
        fig = plt.figure(figsize=(9.0,9.0))
        #fig.suptitle('Vertically stacked subplots')

        ax1 = plt.subplot(3, 2, 1, projection=proj, title="2009 RMSE: 0.0962 W m$^{-2}$")
        ax1.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax2 = plt.subplot(3, 2, 2, projection=proj, title="2009 Bias: -0.00155 W m$^{-2}$")
        ax2.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax3 = plt.subplot(3, 2, 3, projection=proj, title="2015 RMSE: 0.0976 W m$^{-2}$")
        ax3.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax4 = plt.subplot(3, 2, 4, projection=proj, title="2015 Bias: -0.00643 W m$^{-2}$")
        ax4.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax5 = plt.subplot(3, 2, 5, projection=proj, title="2020 RMSE: 0.0975 W m$^{-2}$")
        ax5.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax6 = plt.subplot(3, 2, 6, projection=proj, title="2020 Bias: -0.00627 W m$^{-2}$")
        ax6.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        vmin1 = min([up_flux_2009_rmse.min(), up_flux_2015_rmse.min(), up_flux_2020_rmse.min()])
        vmax1 = max([up_flux_2009_rmse.max(), up_flux_2015_rmse.max(), up_flux_2020_rmse.max()])
        vmin2 = min([up_flux_2009_bias.min(), up_flux_2015_bias.min(), up_flux_2020_bias.min()])
        vmin2 = -0.15
        vmax2 = max([up_flux_2009_bias.max(), up_flux_2015_bias.max(), up_flux_2020_bias.max()])
        c1 = ax1.tricontourf(x,y,up_flux_2009_rmse, vmin=vmin1, vmax=vmax1)
        c2 = ax2.tricontourf(x,y,up_flux_2009_bias,vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        c3 = ax3.tricontourf(x,y,up_flux_2015_rmse,vmin=vmin1, vmax=vmax1)
        c4 = ax4.tricontourf(x,y,up_flux_2015_bias, vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        c5 = ax5.tricontourf(x,y,up_flux_2020_rmse,vmin=vmin1, vmax=vmax1)
        c6 = ax6.tricontourf(x,y,up_flux_2020_bias,vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        #ax1.annotate("flux up")
        #ax2.annotate("flux down")

        fig.colorbar(c5, ax=(ax1,ax3,ax5), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.25,1.0))

        fig.colorbar(c6, ax=(ax2,ax4,ax6), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.7,1.0), ticks=(-0.15,-0.09,-0.03,0.03,0.09,0.15))

    elif False:
        # Downwelling flux
        fig = plt.figure(figsize=(9.0,9.0))
        #fig.suptitle('Vertically stacked subplots')

        ax1 = plt.subplot(3, 2, 1, projection=proj, title="2009 RMSE: 0.852 W m$^{-2}$")
        ax1.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax2 = plt.subplot(3, 2, 2, projection=proj, title="2009 Bias: -0.00695 W m$^{-2}$")
        ax2.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax3 = plt.subplot(3, 2, 3, projection=proj, title="2015 RMSE: 0.0894 W m$^{-2}$")
        ax3.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax4 = plt.subplot(3, 2, 4, projection=proj, title="2015 Bias: -0.0185 W m$^{-2}$")
        ax4.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax5 = plt.subplot(3, 2, 5, projection=proj, title="2020 RMSE: 0.0976 W m$^{-2}$")
        ax5.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax6 = plt.subplot(3, 2, 6, projection=proj, title="2020 Bias: -0.0298 W m$^{-2}$")
        ax6.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        vmin1 = min([down_flux_2009_rmse.min(), down_flux_2015_rmse.min(), down_flux_2020_rmse.min()])
        vmax1 = max([down_flux_2009_rmse.max(), down_flux_2015_rmse.max(), down_flux_2020_rmse.max()])
        vmin2 = min([down_flux_2009_bias.min(), down_flux_2015_bias.min(), down_flux_2020_bias.min()])
        #vmin2 = -0.15
        vmax2 = max([down_flux_2009_bias.max(), down_flux_2015_bias.max(), down_flux_2020_bias.max()])
        c1 = ax1.tricontourf(x,y,down_flux_2009_rmse, vmin=vmin1, vmax=vmax1)
        c2 = ax2.tricontourf(x,y,down_flux_2009_bias,vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        c3 = ax3.tricontourf(x,y,down_flux_2015_rmse,vmin=vmin1, vmax=vmax1)
        c4 = ax4.tricontourf(x,y,down_flux_2015_bias, vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        c5 = ax5.tricontourf(x,y,down_flux_2020_rmse,vmin=vmin1, vmax=vmax1)
        c6 = ax6.tricontourf(x,y,down_flux_2020_bias,vmin=vmin2, vmax=vmax2, levels=10, cmap='coolwarm', norm=colors.CenteredNorm())
        #ax1.annotate("flux up")
        #ax2.annotate("flux down")

        fig.colorbar(c5, ax=(ax1,ax3,ax5), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.25,1.0))

        fig.colorbar(c6, ax=(ax2,ax4,ax6), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.7,1.0))#, ticks=(-0.15,-0.09,-0.03,0.03,0.09,0.15))
    else:
        # Heating rate
        fig = plt.figure(figsize=(9.0,9.0))
        #fig.suptitle('Vertically stacked subplots')

        ax1 = plt.subplot(3, 2, 1, projection=proj, title="2009 RMSE: 0.0225 K day$^{-1}$")
        ax1.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax2 = plt.subplot(3, 2, 2, projection=proj, title="2009 Bias: -0.000907 K day$^{-1}$")
        ax2.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax3 = plt.subplot(3, 2, 3, projection=proj, title="2015 RMSE: 0.0235 K day$^{-1}$")
        ax3.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax4 = plt.subplot(3, 2, 4, projection=proj, title="2015 Bias: -0.00312 K day$^{-1}$")
        ax4.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax5 = plt.subplot(3, 2, 5, projection=proj, title="2020 RMSE: 0.0262 K day$^{-1}$")
        ax5.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        ax6 = plt.subplot(3, 2, 6, projection=proj, title="2020 Bias: -0.00540 K day$^{-1}$")
        ax6.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)

        vmin1 = min([heating_rate_2009_rmse.min(), heating_rate_2015_rmse.min(), heating_rate_2020_rmse.min()])
        vmax1 = max([heating_rate_2009_rmse.max(), heating_rate_2015_rmse.max(), heating_rate_2020_rmse.max()])
        vmin2 = min([heating_rate_2009_bias.min(), heating_rate_2015_bias.min(), heating_rate_2020_bias.min()])
        #vmin2 = -0.15
        vmax2 = max([heating_rate_2009_bias.max(), heating_rate_2015_bias.max(), heating_rate_2020_bias.max()])
        c1 = ax1.tricontourf(x,y,heating_rate_2009_rmse, vmin=vmin1, vmax=vmax1)
        c2 = ax2.tricontourf(x,y,heating_rate_2009_bias,vmin=vmin2, vmax=vmax2, cmap='coolwarm', levels=10, norm=colors.CenteredNorm())
        c3 = ax3.tricontourf(x,y,heating_rate_2015_rmse,vmin=vmin1, vmax=vmax1)
        c4 = ax4.tricontourf(x,y,heating_rate_2015_bias, vmin=vmin2, vmax=vmax2, cmap='coolwarm', levels=10, norm=colors.CenteredNorm())
        c5 = ax5.tricontourf(x,y,heating_rate_2020_rmse,vmin=vmin1, vmax=vmax1)
        c6 = ax6.tricontourf(x,y,heating_rate_2020_bias,vmin=vmin2, vmax=vmax2, cmap='coolwarm', levels=10, norm=colors.CenteredNorm())
        #ax1.annotate("flux up")
        #ax2.annotate("flux down")

        fig.colorbar(c5, ax=(ax1,ax3,ax5), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.25,1.0))

        fig.colorbar(c2, ax=(ax2,ax4,ax6), location='bottom', shrink=0.95, fraction=0.05, pad=0.05, anchor=(0.7,1.0)) #, ticks=(-0.15,-0.09,-0.03,0.03,0.09,0.15))

    dt_2009.close()
    dt_2015.close()
    dt_2020.close()
    dt_geography.close()
    plt.show()


if __name__ == "__main__":
    if False:
        # Plots training and validation vs. epoch
        file_names_validation = ["log_sw.v1.v1.1.txt","log_sw.v1.v1.2.txt","log_sw.v1.v1.3.txt","log_sw.v1.v1.4.txt","log_sw.v1.v1.5.txt", "log_sw.v1.v1.6.txt", "log_sw.v1.v1.7.txt"]
        epochs_1, errors_1 = get_data(file_names_validation, is_training=False, error_name="Loss")
        file_names_training = ["log_sw.v1.v1.0.r_training.txt", "log_sw.v1.v1.2.r_training.txt",]
        epochs_2, errors_2 = get_data(file_names_training, is_training=True, error_name="Loss")

        fig = plt.figure()

        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.set_title("Training of Open Box Neural Network")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (RMSE)")
        ax.set_xlim(140.0,660.0)
        ax.set_ylim(0.0,0.1)

        ax.plot(epochs_1, errors_1) #, lw="1.0", marker=".")
        ax.plot(epochs_2, errors_2) #, lw="1.0", marker=".")

        ax.axvline(647.0, color='red')
        ax.axvline(200.0, color='blue', linestyle='dashed')
        ax.legend(("Loss on validation dataset","Loss on training dataset","Training stopping point","Change in loss weighting"))

        ax.axvline(285.0, color='blue', linestyle='dashed')
        ax.axvline(360.0, color='blue', linestyle='dashed')
        ax.axvline(515.0, color='blue', linestyle='dashed')

        #ax.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
        #ax.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
        #ax.step(epochs_1, errors_1, where = 'pre', label = 'vert_first')
        #ax.step(epochs_1, errors_1, where = 'post', label = 'flat_first')
        if False:
            stem_1 = plt.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
            stem_2 = plt.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
            # stem[1] change stemlines
            stem_1[1].set_linewidth(0.5)
            stem_2[1].set_linewidth(0.5)
        plt.show()
    elif False:
        # Plots training and validation vs. epoch with no weight change at 360
        file_names_validation = ["log_sw.v1.v1.1.txt","log_sw.v1.v1.2.txt","log_sw.v1.v1.3.txt","log_sw.v1.v1b.1.txt"]
        epochs_1, errors_1 = get_data(file_names_validation, is_training=False, error_name="Loss")
        file_names_training = ["log_sw.v1.v1.0.r_training.txt", "log_sw.v1.v1.2.r_training.txt",]
        epochs_2, errors_2 = get_data(file_names_training, is_training=True, error_name="Loss")

        fig = plt.figure()

        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.set_title("Training of Open Box Neural Network")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (RMSE)")
        ax.set_xlim(140.0,660.0)
        ax.set_ylim(0.0,0.1)

        ax.plot(epochs_1, errors_1) #, lw="1.0", marker=".")
        ax.plot(epochs_2, errors_2) #, lw="1.0", marker=".")

        ax.axvline(647.0, color='red')
        ax.axvline(200.0, color='blue', linestyle='dashed')
        ax.legend(("Weighted Validation Loss","Weighted Training Loss","Training Stopping Point","Change in Loss Weighting"))

        ax.axvline(285.0, color='blue', linestyle='dashed')
        ax.axvline(360.0, color='blue', linestyle='dashed')
        ax.axvline(515.0, color='blue', linestyle='dashed')

        #ax.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
        #ax.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
        #ax.step(epochs_1, errors_1, where = 'pre', label = 'vert_first')
        #ax.step(epochs_1, errors_1, where = 'post', label = 'flat_first')
        if False:
            stem_1 = plt.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
            stem_2 = plt.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
            # stem[1] change stemlines
            stem_1[1].set_linewidth(0.5)
            stem_2[1].set_linewidth(0.5)
        plt.show()
    elif True:
        # Plots validation flux and heating rate validation vs. epoch
        is_aggressive = True
        if not is_aggressive:
            file_names_validation = ["log_sw.v1.v1.1.txt","log_sw.v1.v1.2.txt","log_sw.v1.v1.3.txt","log_sw.v1.v1.4.txt","log_sw.v1.v1.5.txt", "log_sw.v1.v1.6.txt", "log_sw.v1.v1.7.txt"] #Original data
        else:
            file_names_validation = ["log_sw.v1.v1.1.txt","log_sw.v1.v1.2.txt","log_sw.v1.v1.3.txt","log_sw.v1.v1a.1.txt",]  # Aggressive Model
        #file_names_validation = ["log_sw.v1.v1.1.txt","log_sw.v1.v1.2.txt","log_sw.v1.v1.3.txt","log_sw.v1.v1b.1.txt"]   # No change in weighting
        #epochs_1, errors_1 = get_data(file_names_validation, is_training=False, error_name="Full Flux Loss")

        epochs_1c, errors_1c = get_data(file_names_validation, is_training=False, error_name="Direct Flux Loss")

        epochs_1d, errors_1d = get_data(file_names_validation, is_training=False, error_name="Diffuse Flux Loss")

        epochs_1a, errors_1a = get_data(file_names_validation, is_training=False, error_name="Direct Flux Loss")
        
        epochs_1b, errors_1b = get_data(file_names_validation, is_training=False, error_name="Diffuse Flux Loss")

        epochs_2, errors_2 = get_data(file_names_validation, is_training=False, error_name="Full Heating rate Loss")

        epochs_3, errors_3 = get_data(file_names_validation, is_training=False, error_name="Direct Heating Rate Loss")

        epochs_4, errors_4 = get_data(file_names_validation, is_training=False, error_name="Diffuse Heating Rate Loss")

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        #ax.grid('both')
       
        ax.set_xlim(140.0,660.0)
        ax.set_ylim(0.0,0.2)
        ax.set_title("Unweighted losses on the validation test set")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (RMSE)")

        ax.plot(epochs_1c, errors_1c) 
        ax.plot(epochs_1d, errors_1d) 
        #ax.plot(epochs_1a, errors_1a) 
        #ax.plot(epochs_1b, errors_1b) 
        #ax.plot(epochs_2, errors_2) 
        ax.plot(epochs_3, errors_3) 
        ax.plot(epochs_4, errors_4) 

        if not is_aggressive:
            ax.axvline(647.0, color='red')
        #ax.axvline(419.0, color='red')
        ax.axvline(200.0, color='blue', linestyle='dashed')
        if not is_aggressive:
            ax.legend(("Direct flux loss", "Diffuse flux loss",#"Heating Rate Loss",
                   "Direct extinction loss", "Diffuse heating rate loss", 
                   "Training stopping point", 
                   "Change in loss weighting"))
        else:
            ax.legend(("Direct flux loss", "Diffuse flux loss",#"Heating Rate Loss",
                        "Direct extinction loss", "Diffuse heating rate loss", 
                   "Change in loss weighting"))
        #ax.legend(("Direct Flux Loss","Diffuse Flux Loss","Heating Rate Loss","Direct HR", "Diffuse HR", "Training Stopping Point","Change in Loss Weighting"))

        ax.axvline(285.0, color='blue', linestyle='dashed')
        ax.axvline(360.0, color='blue', linestyle='dashed')
        if not is_aggressive:
            ax.axvline(515.0, color='blue', linestyle='dashed')

        #ax.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
        #ax.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
        #ax.step(epochs_1, errors_1, where = 'pre', label = 'vert_first')
        #ax.step(epochs_1, errors_1, where = 'post', label = 'flat_first')
        if False:
            stem_1 = plt.stem(epochs_1, errors_1, markerfmt='.', linefmt="C0-")
            stem_2 = plt.stem(epochs_2, errors_2, markerfmt='.', linefmt="C1-")
            # stem[1] change stemlines
            stem_1[1].set_linewidth(0.5)
            stem_2[1].set_linewidth(0.5)
        plt.show()
    elif False:
        # Vertical error profile
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        if False:
            file_names = ["layer_hr_error_2009.txt","layer_hr_error_2015.txt", "layer_hr_error_2020.txt"]
            ax.set_title("Vertical Distribution of Heating Rate RMSE")
            ax.set_xlabel("Heating Rate RMSE in K day$^{-1}$")
            ax.set_ylabel("Atmospheric Layer")
        elif True:
            file_names = ["layer_flux_down_error_2009.txt","layer_flux_down_error_2015.txt", "layer_flux_down_error_2020.txt"]
            ax.set_title("Vertical Distribution of Downwelling Flux RMSE")
            ax.set_xlabel("Downwelling Flux RMSE in W m$^{-2}$")
            ax.set_ylabel("Atmospheric Layer")
        else:
            file_names = ["layer_flux_up_error_2009.txt","layer_flux_up_error_2015.txt", "layer_flux_up_error_2020.txt"]
            ax.set_title("Vertical Distribution of Upwelling Flux RMSE")
            ax.set_xlabel("Upwelling Flux RMSE in W m$^{-2}$")
            ax.set_ylabel("Atmospheric Layer")
       
        ax.set_xlim(-0.0,0.2)
        ax.invert_yaxis()
        ax.grid('both')
        for file_name in file_names:
            indices, errors = get_layer_data(file_name, is_rmse=True)
            ax.plot(errors, indices)
        ax.legend(('2009','2015','2020'))    
        plt.show()

    elif True:

        plot_geographic_results()

    else:
        # analyzes Ukkonen's validation vs. epoch
        #file_name = "/home/hws/src/RTE-RRTMGP-NN/examples/emulator-training/scripts/log_ukkonen_ecrad_homo_v1.txt"
        file_name = "/home/hws/src/RTE-RRTMGP-NN/examples/emulator-training/scripts/log_ukkonen_ecrad_homo_composite.txt"
        #file_name = "/home/hws/src/RTE-RRTMGP-NN/examples/emulator-training/scripts/log_ukkonen_ecrad_v2.3.txt"
        epochs, val_loss, train_loss = generate_epoch_error_ukkonen(file_name, is_best_in_25=False)

        fig = plt.figure()

        ax = fig.add_axes([0.18, 0.18, 0.64, 0.64])
       
        ax.set_xlim(140.0, 575.0)
        ax.set_ylim(0.0,0.0002)
        ax.set_title("Training of (Ukkonen, 2022a)(Ukkonen, 2022b)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (RMSE)")

        ax.plot(epochs, val_loss) 
        ax.plot(epochs, train_loss) 
        ax.axvline(572.0, color='red')

        ax.legend(('Validation Loss','Training Loss','Training Stopping Point'))  

        plt.show()

        

            


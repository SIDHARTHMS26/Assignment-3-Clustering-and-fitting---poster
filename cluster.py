# Importing necassery libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import error as err

# defining the read function for reading all csv files and doing the cleaning and transposing of datafile.
def read_file(file_name):
    """
    This function reads the file from given address, processes the input,
    and returns a normal dataframe and transposed one
    
    Parameters
    ----------
    file_name : string
    Name of the file to be read, including the full address.
    
    Returns
    -------
    df_changed : Dataframe
    File content read into dataframe and preprocessed.
    df_transposed : Dataframe
    File content read into dataframe, preprocessed and transposed.
    
    """
    df = pd.read_csv(file_name)
    df_changed = df.drop(
        columns=["Series Name", "Series Code", "Country Code"])
    countries = ['United States', 'Japan']
    # Remove the string of year from column names,Ecxtracting the data transposing it and renaming it.
    df_changed.columns = df_changed.columns.str.replace(
        ' \[YR\d{4}\]', '', regex=True)
    df_changed = df_changed[df_changed['Country Name'].isin(countries)].T
    df_changed = df_changed.rename({'Country Name': 'year'})
    df_changed = df_changed.reset_index().rename(columns={'index': 'year'})
    df_changed.columns = df_changed.iloc[0]
    df_changed = df_changed.iloc[1:]
    df_changed["year"] = pd.to_numeric(df_changed["year"])
    df_changed['United States'] = pd.to_numeric(df_changed['United States'])
    df_changed['Japan'] = pd.to_numeric(df_changed['Japan'])
    return df_changed


# defining the curve function.
def curve_fun(t, scale, growth):
    """
 
    Parameters
    ----------
    t : TYPE
    List of values
    scale : TYPE
    Scale of curve.
    growth : TYPE
    Growth of the curve.
    Returns
    -------
    c : TYPE
    Result
    """
    c = scale * np.exp(growth * (t-1960))
    return c


# fucntion to create the linplot
def total_methane(data_3):
    """
    plot the total Methane emission of this two countries.
    """
    # line plot of total methane emission of us and japan
    plt.plot(data_3["year"], data_3["United States"])
    plt.plot(data_3["year"], data_3["Japan"])
    plt.xlim(1990, 2018)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Methane emission in (kt co2 eqlnt)", fontsize=12)
    plt.legend(['US', 'JPN'])
    plt.title("Total Methane emission of \nUS & Japan in kt of co2 eqlnt ", fontsize=12)
    plt.savefig("totalCH4.png", dpi = 300, bbox_inches='tight')
    plt.show()
 
    
# read the csv file and and store it in to a dataFrame
data_1 = read_file('energy_meth.csv')
data_2 = read_file('Agri_meth.csv')
data_3 = read_file('total_methane.csv')


"""
plotting the Curve fit and Error fit of MEthane emission in energy sector and doing the prediction to2030 of United States.
"""
# Doing the curve Fit for US
param, cov = opt.curve_fit(curve_fun, data_1["year"], data_1["United States"], p0=[4e8,
                                                                                   0.1])
sigma = np.sqrt(np.diag(cov))

#Doing Error fit for the Selected Countries.
low, up = err.err_ranges(data_1["year"], curve_fun, param, sigma)
data_1
data_1["fit_value"] = curve_fun(data_1["year"], * param)

#Plotting the co2 emission values for US
plt.figure()
plt.title("Methane Emision in Energy Sector - United States", fontsize=12)
plt.plot(data_1["year"], data_1["United States"], label="data")
plt.plot(data_1["year"], data_1["fit_value"], c="red", label="fit")
plt.fill_between(data_1["year"], low, up, alpha=0.2)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Methane Emision in Energy Sector\n(kmt of co2 eqltn)", fontsize=12)
plt.savefig("US_curve.png", dpi = 300, bbox_inches='tight')
plt.show()

# Plotting the predicted values for US
plt.figure()
plt.title("US Methane emission prediction", fontsize=12)
pred_year = np.arange(1990, 2030)
pred_ind = curve_fun(pred_year, *param)
plt.plot(data_1["year"], data_1["United States"], label="data")
plt.plot(pred_year, pred_ind, label="predicted values")
plt.legend()
plt.xlabel("Year", fontsize=12)
plt.ylabel("Methane Emission in Energy Sector\n(kmt CO2 eqlnt)", fontsize=12)
plt.savefig("US_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()


"""
plotting the Curve fit and Error fit of MEthane emission in energy sector and doing the  prediction to 2030 of japan.
"""
# Doing the curve Fit for US
param, cov = opt.curve_fit(curve_fun, data_1["year"], data_1["Japan"], p0=[4e8,
                                                                           0.1])
sigma = np.sqrt(np.diag(cov))

#Doing Error fit for the Selected Countries.
low, up = err.err_ranges(data_1["year"], curve_fun, param, sigma)
data_1
data_1["fit_value"] = curve_fun(data_1["year"], * param)

#Plotting the enrgy sector methane emission values for japan
plt.figure()
plt.title("Methane Emision in Energy Sector - japan", fontsize=12)
plt.plot(data_1["year"], data_1["Japan"], label="data")
plt.plot(data_1["year"], data_1["fit_value"], c="red", label="fit")
plt.fill_between(data_1["year"], low, up, alpha=0.2)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Methane Emision in Energy Sector\n(kmt of co2 eqltn)", fontsize=12)
plt.savefig("E_japan_curve.png", dpi = 300, bbox_inches='tight')
plt.show()

# Plotting the predicted values for japan
plt.figure()
plt.title("japan Methane emission prediction", fontsize=12)
pred_year = np.arange(1990, 2030)
pred_ind = curve_fun(pred_year, *param)
plt.plot(data_1["year"], data_1["Japan"], label="data")
plt.plot(pred_year, pred_ind, label="predicted values")
plt.legend()
plt.xlabel("Year", fontsize=12)
plt.ylabel("Methane Emission in Energy Sector\n(kmt CO2 eqlnt)", fontsize=12)
plt.savefig("E_japan_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()

# calling th eline plot
total_methane(data_3)
"""
scatter plot using Kmeans Algorithm
"""
# plotting scatter plot of the US and Japan's Methane Emission in Energy Sector
kmean = cluster.KMeans(n_clusters=2).fit(data_1)
label = kmean.labels_
plt.scatter(data_1["United States"], data_1["Japan"], c=label, cmap="jet")
plt.title("US and Japan Methane Emission in Energy Sector", fontsize=12)
plt.xlabel("United States", fontsize=12)
plt.ylabel("Japan", fontsize=12)
c = kmean.cluster_centers_
plt.savefig("Scatter_us_and_japan.png", dpi = 300, bbox_inches='tight')
plt.show()

# Here we cretes a datafrme and plotting a scatter plot of Japan in both Energy and Agricultural Methane Emission
japan = pd.DataFrame()
japan["Energy_methane"] = data_1["Japan"]
japan["Agricultural_methane"] = data_2["Japan"]
kmean = cluster.KMeans(n_clusters=2).fit(japan)
label = kmean.labels_
plt.scatter(japan["Energy_methane"],
            japan["Agricultural_methane"], c=label, cmap="jet")
plt.title("Energy Sector Methane Emission \n& Agricultural Methane Emission -Japan", fontsize=12)
plt.xlabel("Energy Sector Methane Emission", fontsize=12)
plt.ylabel("Agricultural Methane Emission", fontsize=12)
plt.savefig("Scatter_CO2_vs_Renewable_India.png", dpi=300, bbox_inches='tight')
c = kmean.cluster_centers_
for t in range(2):
    xc, yc = c[t, :]
    plt.plot(xc, yc, "ok", markersize=8)

plt.savefig("Scatter_japan.png", dpi = 300, bbox_inches='tight')
plt.show()






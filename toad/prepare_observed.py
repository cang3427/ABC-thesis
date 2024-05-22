import os, sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

DATA_DIR = "./toad/data"

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        sys.exit("Error: Directory does not exist. This directory is required, as it should contain the collected data.")     
    
    # Loading tracking data
    toad_data_2009 = pd.read_csv(os.path.join(DATA_DIR, "radio2009.csv"))
    toad_data_2010 = pd.read_csv(os.path.join(DATA_DIR, "radio2010.csv"))    
    toad_data_2009["Toad"] = toad_data_2009["Toad"].astype('str') + "_09"    
    toad_data_2010["Toad"] = toad_data_2010["Toad"].astype('str') + "_10"    
    toad_data = pd.concat([toad_data_2009, toad_data_2010])
    toad_data = toad_data[toad_data.Cycle == "day"][["Toad", "Date", "Easting", "Northing"]]
    
    # Loading waterline coordinate data
    wline = gpd.read_file(os.path.join(DATA_DIR, "Final Waterline.shp"))
    coords = []
    for geom in wline.geometry:
        coords += list(geom.coords)
        
    coords_east = [[coord[0]] for coord in coords]
    coords_north = [[coord[1]] for coord in coords]
    
    # Determining approximate waterline axis
    wline_model = LinearRegression().fit(coords_east, coords_north)
    slope = wline_model.coef_[0][0]
    
    # Projecting data onto a new coordinate system with the waterline as the x axis
    rotation_angle = math.atan(-slope)
    rotation_matrix = np.array([[math.cos(rotation_angle), math.sin(rotation_angle)],
                               [-math.sin(rotation_angle), math.cos(rotation_angle)]])
    
    centre = np.array(toad_data[["Easting", "Northing"]].mean())
    centred_coords = np.column_stack((toad_data["Easting"], toad_data["Northing"])) - centre
    
    aligned_coords = np.linalg.multi_dot([centred_coords, rotation_matrix])
    
    # Update data with new coords
    toad_data[["x", "y"]] = aligned_coords

    # Removing toads with only 1 day of tracked movement
    toad_data = toad_data[toad_data.duplicated(subset = ["Toad"], keep = False)]
    
    # Removing duplicate observations for same Toad and date
    toad_data = toad_data.drop_duplicates(subset=["Toad", "Date"])

    # Convert toad IDs to indices
    toad_data["Toad"] = pd.factorize(toad_data["Toad"])[0] + 1
    
    # Create day column for each toad starting from day 1
    toad_data["Date"] = pd.to_datetime(toad_data["Date"])
    toad_data["Day"] = (toad_data["Date"] - toad_data.groupby("Toad")["Date"].transform("min")).dt.days + 1
    
    # Convert dataframe to day x toad matrix, with x positions as entries
    toad_day_matrix = toad_data.pivot_table(index = "Day", columns = "Toad", values = "x").to_numpy()
    np.save(os.path.join(DATA_DIR, "observed_data.npy"), toad_day_matrix)

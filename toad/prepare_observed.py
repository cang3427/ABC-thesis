import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from toad_utils import summarise_sample

DATA_DIR = "./model_choice/toad/data"

if __name__ == "__main__":
    
    # Loading tracking data
    toadData2009 = pd.read_csv(os.path.join(DATA_DIR, "radio2009.csv"))
    toadData2010 = pd.read_csv(os.path.join(DATA_DIR, "radio2010.csv"))    
    toadData2009["Toad"] = toadData2009["Toad"].astype('str') + "_09"    
    toadData2010["Toad"] = toadData2010["Toad"].astype('str') + "_10"    
    toadData = pd.concat([toadData2009, toadData2010])
    toadData = toadData[toadData.Cycle == "day"][["Toad", "Date", "Easting", "Northing"]]
    
    # Loading waterline coordinate data
    wline = gpd.read_file(os.path.join(DATA_DIR, "Final Waterline.shp"))
    coords = []
    for geom in wline.geometry:
        coords += list(geom.coords)
        
    coords_east = [[coord[0]] for coord in coords]
    coords_north = [[coord[1]] for coord in coords]
    
    # Determining approximate waterline axis
    wlineModel = LinearRegression().fit(coords_east, coords_north)
    slope = wlineModel.coef_[0][0]
    
    # Projecting data onto a new coordinate system with the waterline as the x axis
    rotationAngle = math.atan(-slope)
    rotationMatrix = np.array([[math.cos(rotationAngle), math.sin(rotationAngle)], \
                            [-math.sin(rotationAngle), math.cos(rotationAngle)]])
    
    centre = np.array(toadData[["Easting", "Northing"]].mean())
    centredCoords = np.column_stack((toadData["Easting"], toadData["Northing"])) - centre
    
    aligned_coords = np.linalg.multi_dot([centredCoords, rotationMatrix])
    
    # Update data with new coords
    toadData[["x", "y"]] = aligned_coords

    # Removing toads with only 1 day of tracked movement
    toadData = toadData[toadData.duplicated(subset = ["Toad"], keep = False)]
    
    # Removing duplicate observations for same Toad and date
    toadData = toadData.drop_duplicates(subset=["Toad", "Date"])

    # Convert toad IDs to indices
    toadData["Toad"] = pd.factorize(toadData["Toad"])[0] + 1
    
    # Create day column for each toad starting from day 1
    toadData["Date"] = pd.to_datetime(toadData["Date"])
    toadData["Day"] = (toadData["Date"] - toadData.groupby("Toad")["Date"].transform("min")).dt.days + 1
    
    # Convert dataframe to day x toad matrix, with x positions as entries
    toadDayMatrix = toadData.pivot_table(index = "Day", columns = "Toad", values = "x").to_numpy()
    np.save(os.path.join(DATA_DIR, "observed_data.npy"), toadDayMatrix)

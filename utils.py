import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
import csv
import numpy as np
import argparse

def plot_2D_points(points, values, xlim, ylim, x_label, y_label, title, path, name, vmin, vmax,has_vmin=True, has_vmax=True):
    '''
    points: shape (num_poins,2)
    values: shape (num_poins,)
    xlim: [lb,ub] lower bound and upper bound for x
    ylim: [lb,ub] lower bound and upper bound for y
    x_label: oAo
    y_label: oAo
    vmin: min value of a point
    vmax: max value of a point
    path: where to save the image
    '''
    plt.figure(figsize=(7,7))
    if has_vmin and has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmin=vmin, vmax=vmax)
    elif not has_vmin and has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmax=vmax)
    elif has_vmin and not has_vmax:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet', vmin=vmin)
    else:
        plt.scatter(points[:,0], points[:,1], c=values, cmap='jet')
    if has_vmin or has_vmax:
        plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim); plt.ylim(ylim)
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{name}.png")
    plt.show()
    
def plot_xy_curves(x, ys_list, labels_list, xlim, ylim, x_label, y_label, title, path, name, vis=False):
    '''
    x: 1D points
    ys_list: list of ys, shape: (num_curves, num_points_per_curve)
    xlim: [lb,ub] lower bound and upper bound for x
    ylim: [lb,ub] lower bound and upper bound for y
    x_label: oAo
    y_label: oAo
    path: where to save the image
    '''
    plt.figure(figsize=(7,7))
    for i, ys in enumerate(ys_list):
        if len(labels_list)!=0:
            plt.plot(x, ys, label=labels_list[i])
            plt.scatter(x,ys)
        else:
            plt.plot(x, ys)
            plt.scatter(x,ys)
    plt.title(title)
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if xlim!=None and ylim!=None:
        plt.xlim(xlim); plt.ylim(ylim)
    plt.legend()
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{name}.png")
    if vis:
        plt.show()
    
def get_2D_grid_points(num_points, xy_lower_bound, xy_upper_bound):
    x_ax = np.linspace(start=xy_lower_bound[0], stop=xy_upper_bound[0], num=(int)(np.sqrt(num_points)))
    y_ax = np.linspace(start=xy_lower_bound[1], stop=xy_upper_bound[1], num=(int)(np.sqrt(num_points)))
    x_grid, y_grid = np.meshgrid(x_ax, y_ax)
    grid = np.concatenate((x_grid[:,:,np.newaxis], y_grid[:,:,np.newaxis]), axis=-1)
    grid_points = grid.reshape((-1,2)) # points in 2D
    return grid_points

def load_csv_data(data_path, preppend_one=True, remove_first_row=False):
    '''
    load csv data into a np array of type float, excluding the column names (first row)
    '''
    data = []
    with open(data_path, mode ='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data.append(line)
    if remove_first_row:
        data.pop(0)
    data = np.array(data)
    data = data.astype(np.float64)
    # preppend 1 to the end of each feature vector (bias trick)
    if preppend_one:
        one = np.ones((len(data), 1), dtype=np.float64)
        data = np.concatenate((one, data), axis=1)
    return data

def write_csv_row(file_path, row):
    '''
    fill out a row for the csv file
    '''
    with open(file_path, 'a', newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(row)
        
def write_csv_rows(file_path, rows):
    '''
    fill out multiple rows for the csv file
    '''
    with open(file_path, 'a', newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerows(rows) 

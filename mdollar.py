# Python $M Mid-Air Gesture Recognizer
# 
# This file contains a Python implementation of the $M algorithm.
# The material used can be found online at: [TO BE ADDED]
#
# The academic publication for the $M recognizer, and what should be
# used to cite it, is:
#
# [TO BE ADDED]
#
# This software is distributed under the "New BSD License" agreement:
#
# Copyright (C) 2024, Anonymous
# All rights reserved. Last updated April 20, 2024.

from math import pi, atan2, cos, sin, inf

# momona added these 
import numpy as np
# from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
import pickle

RESAMPLE_SIZE = 64
ORIGIN = (0, 0)
SQUARE_SIZE = 1
ANGLE_RANGE = (2 / 180) * pi
ANGLE_PRECISION = (2 / 180) * pi
PHI = 0.5 * (-1.0 + (5.0)**0.5)
N_PC = 50

class Mdollar:
    """
    template input should be gesture name x data
    where data contains N types of biosignal data
    and each biosignal data has C channels
    """
    def __init__(self,templates):
        # format the example gestures
        self.unistrokes = []
        for template in templates:
            stroke = Stroke(template[1])
            self.unistrokes.append(stroke)
            self.unistrokes[-1].name = template[0]
            self.unistrokes[-1].pca = stroke.apply_pca(N_PC)

    def get_gesture(self, points):
        stroke = Stroke(points)

        # search for the closest gesture (ie. with minimal distance)
        min_distance = inf
        gesture_name = ''
        for template_stroke in self.unistrokes:
            stroke.apply_pca(N_PC=N_PC,pca=template_stroke.pca)
            
            # flatten the biosignal which is now time (64) x # PCs 
            template_stroke.flatten_list()
            stroke.flatten_list()

            # make a list of template types and min distances for each template type to candidate gesture
            distance = stroke.path_distance(template_stroke.points)
            if distance < min_distance:
                # update the current best gesture
                min_distance = distance
                gesture_name = template_stroke.name
            # at the end, sort the list and return the order number of the templates 
            # in the other file, I can pick which order number corresponds to the actual candidate 
        return gesture_name
        


class Stroke:

    def __init__(self, biosignals, should_format=False):
        self.biosignals = biosignals
        # print(len(self.biosignals),len(self.biosignals[0]))
        if should_format:
            for btype in range(len(self.biosignals)):
                self.btype = btype
                # also compute standard deviation of entire dataset in the biosignal type
                all_sigs = [] 
                for channel in range(len(self.biosignals[btype])):
                    self.channel = channel
                    self.points = self.biosignals[btype][channel]
                    self.resample() # just resample first
                    self.demean() # demean the channel
                    all_sigs.extend(self.biosignals[btype][channel]) # for calculating std of entire biosignal
                self.std = np.std(np.asarray(all_sigs)) # calculate std of entire biosignal # TODO NEED TO REWRITE THIS TO BE POINTWISE
                for channel in range(len(self.biosignals[btype])):
                    self.channel = channel
                    self.points = self.biosignals[btype][channel]
                    self.scale_to() # scale to std

    def resample(self):
        points = self.points

        N = len(points) # length of original data

        # get old x
        old_x = [0]
        for ix in range(1,N):
            old_x.extend([ix])

        # get new x 
        new_x = [0]
        for ix in range(1,RESAMPLE_SIZE):
            new_x.extend([ix/(RESAMPLE_SIZE-1)*(N-1)])

        new_points = [points[0]] # add first point
        ix = 0
        for x in new_x[1:-1]:
            # find closest old_x that has new_x 
            while old_x[ix] < x:
                ix += 1
            y1,y2 = points[ix-1],points[ix]
            x1,x2 = old_x[ix-1],old_x[ix]
            # interpolate between the two old points
            m = (y2-y1)/(x2-x1)
            b = y2 - m*x2
            new_point = m*x + b
            new_points.extend([new_point])
        new_points.extend([points[-1]]) # add last point
        self.points = new_points
        self.biosignals[self.btype][self.channel] = self.points

    def demean(self):
        # calculate mean of entire dataset
        mean_points = 0
        for p in self.points:
            mean_points += p
        mean_points = mean_points / len(self.points)
        new_points = [x - mean_points for x in self.points]
        self.points = new_points
        self.biosignals[self.btype][self.channel] = self.points 

    def scale_to(self):
        new_points = []

        for p in self.points:
            new_points.append(
                p / self.std # TODO not sure if this is the correct order
            )
        self.points = new_points
        self.biosignals[self.btype][self.channel] = self.points

    def path_distance(self, points):
        n = len(points)
        return sum([distance(self.points[i], points[i]) / n for i in range(n)])
    
    def apply_pca(self,N_PC,pca=None):
        if len(self.biosignals) > 1:
            points = []
            for ix in range(len(self.biosignals)):
                points += self.biosignals[ix] #+ self.biosignals[1]
        else:
            points = self.biosignals[0]

        if pca is None:
            pca = get_pca(self.biosignals,N_PC)

        self.transformed_biosignals = np.matmul(np.asarray(points).T, pca) 
        return pca

    def flatten_list(self):
        new_points = []
        for biosignal in self.transformed_biosignals.T:
            new_points.extend(biosignal)
        self.points = new_points


def distance(p1, p2):
    return abs(p2 - p1)

# NEW PCA 
def get_pca(data,N_PC):
    if len(data) > 1:
        points = []
        for ix in range(len(data)):
            points += data[ix] #+ self.biosignals[1]
    else:
        points = data[0]

    points = data[0] + data[1] 
    N_time = len(points[0]) 
    N_feat = len(points) 
    
    # THE ACTUAL PCA
    # 1) calculate covariance matrix for standardized data
    covariance_matrix = np.cov(np.asarray(points).T, ddof = 1, rowvar = False) # OLD 
    covs = []
    for j in range(N_feat):
        covariance = []
        for k in range(N_feat):
            total_sum = 0
            for i in range(N_time):
                total_sum += points[j][i] * points[k][i]
            covariance.extend([total_sum / (N_time-1)])
        covs.append(covariance)

    # 2) eigenvalue decomposition -- can't really change this 
    eigenvalues, eigenvectors = np.linalg.eig(covs)

    # 3) sort principal components
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1]

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns

    # 4) calculate explained variance
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    # # 5) reduce data via principal components
    # new_biosignals = np.matmul(np.asarray(points).T, sorted_eigenvectors[:,:N_PC]) # transform the original data

    # # 6) determine explained variance
    # total_explained_variance = sum(explained_variance[:N_PC])
    print(sorted_eigenvectors[:,:N_PC].shape) # 88 x 50
    return sorted_eigenvectors[:,:N_PC]

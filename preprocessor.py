import json
import random
import sys
import pathlib
import os
import pickle

from sklearn.model_selection import train_test_split
import numpy as np

def flag(dir, file_name):
	# Find which time series do not satisfy category 2 requirements

	path = ''.join([dir, file_name]) 
	with open(path, "rb") as file:
		time_stamps = np.load(file, allow_pickle=True)
	
	path = ''.join([dir, "L1_O3a_CAT_2.json"])
	with open(path) as file:
		data_str = file.read()
	segments_a = json.loads(data_str)["segments"]

	path = ''.join([dir, "L1_O3b_CAT_2.json"])
	with open(path) as file:
		data_str = file.read()
	segments_b = json.loads(data_str)["segments"]

	segments = segments_a+segments_b

	flags = np.zeros(len(time_stamps))
	for time_indx in range(len(time_stamps)):

		time_stamp = time_stamps[time_indx]
		segment_indx = 0
		while time_stamp > segments[segment_indx][1] and segment_indx < len(segments)-1:
			segment_indx += 1
		if segments[segment_indx][0] <= time_stamp <= segments[segment_indx][1] and segments[segment_indx][0] <= time_stamp+1 <= segments[segment_indx][1]:
			flags[time_indx] = 2  # good :)
		else:
			flags[time_indx] = 1  # bad :(

	uh_ohs = []
	for i, j in enumerate(flags):
		if j == 1:
			uh_ohs.append(i)
	print(uh_ohs)
	unique, counts = np.unique(flags, return_counts=True)
	print(dict(zip(unique, counts)))
	return uh_ohs

def prep(file_name_base):
	# Shuffle data and write to pickle files.
	series_load_path = ''.join(["data", file_name_base, ".csv"])
	times_load_path = ''.join(["data", file_name_base, "_times.csv"])
	with open(series_load_path, "rb") as file:
		series = np.load(file)
	with open(times_load_path, "rb") as file:
		times = np.load(file)
	perm = np.random.permutation(series.shape[0])
	series_shuffled = series[perm]
	times_shuffled = times[perm]
	print(len(series_shuffled))
	print(len(times_shuffled))
	series_save_path = ''.join([series_load_path[:-4]])
	times_save_path = ''.join([times_load_path[:-4]])
	pathlib.Path(os.path.abspath(series_save_path)).parent.mkdir(parents=True, exist_ok=True)
	pathlib.Path(os.path.abspath(times_save_path)).parent.mkdir(parents=True, exist_ok=True)
	pickle.dump(series_shuffled, open(os.path.abspath(series_save_path), "wb"))
	pickle.dump(times_shuffled, open(os.path.abspath(times_save_path), "wb"))

def check_overlap(file_name, uh_ohs):
	if file_name is not None:
		with open(file_name, "rb") as file:
			indxs = np.load(file, allow_pickle=True)
	else:
		indxs = [x+2000 for x in [1, 5, 8, 13, 22, 24, 27, 29, 30, 33, 36, 40, 43, 48, 50, 59, 66]]
	batches_done = 0
	sketchiness_scores = []
	batch_size = 10
	print(len(indxs))
	for indx_no in range(0, len(indxs), batch_size):
		batch_indxs = []
		for in_batch_counter in range(batch_size):
			batch_indxs.append(indxs[indx_no+in_batch_counter])
		sketchiness_scores.append(sum(x in batch_indxs for x in uh_ohs))
		batches_done += 1
		print(batches_done, " of ", len(indxs), " done")
	print(sketchiness_scores)

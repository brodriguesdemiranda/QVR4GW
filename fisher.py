# This script is for Fisher information calculations

import sys
import os
import argparse
import math
import pickle
import time
import pathlib

import torch
import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

import utils

torch.manual_seed(vars.GLOBAL_SEED)

torch.set_default_tensor_type(torch.DoubleTensor)
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits, shots=256)
#dev = qml.device("default.qubit", wires=2)

time_indexes = []
series_indexes = []
eigs = []
global_fish = np.zeros((24, 24))
fish_counter = 0

initial_alphas_g = torch.tensor([])
initial_mus_g = torch.tensor([])
initial_sigmas_g = torch.tensor([])
initial_eta_0_g = torch.tensor([])

def get_series_training_cycler(Xtr: np.array, n_series_batch: int) -> utils.DataGetter:
	"""
	:param Xtr: Training data consisting of time series', features, and time in the
                    dimensions.
	:param n_series_batch: How many series to fetch for the batch.
	:return x_cycler: An object which iterates through time series'.
	Get a cycler to iterate over time series' which are randomly selected.
	"""
	x_cycler = utils.DataGetter(Xtr, n_series_batch, auto_shuffle=False)
	return x_cycler

def get_timepoint_training_cycler(Xtr: np.array, n_t_batch: int) -> utils.DataGetter:
	"""
	:param Xtr: Training data consisting of time series', features, and time in the
                    dimensions.
	:param n_t_batch: How many time points to fetch for the batch.
	:return t_cycler: An object which iterates through time points.
	Get a cycler to iterate through time points which are randomly selected.
	"""
	n_time_points = Xtr.shape[2]
	T = torch.tensor(np.arange(n_time_points))
	t_cycler = utils.DataGetter(T, n_t_batch, auto_shuffle=True)
	return t_cycler

x_g = None
t_g = None
wires_g = None
k_g = None
embed_func_g = None
transform_func_g = None
diag_func_g = None
observable_g = None
embed_func_params_g = {}
transform_func_params_g = {}
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def fisher(params) -> float:
	alphas = torch.reshape(params[:18], (3, 2, 3))
	sigmas = params[18:21]
	mus = params[21:44]
	D = sample_M(sigmas, mus).detach().numpy()
	embed_func_g(x_g, wires_g, **embed_func_params_g)
	transform_func(alphas, wires_g, **transform_func_params_g)
	diag_func(D*t_g, n_qubits, k=k_g)
	qml.adjoint(transform_func_g)(alphas, wires=range(n_qubits), **transform_func_params_g)
	coeffs = np.ones(len(wires_g))/len(wires_g)
	H = qml.Hamiltonian(coeffs, observable)
	expval = qml.expval(H)
	return expval

@qml.qnode(dev, interface="torch")
def get_anomaly_expec(x: np.array, t: float, D: torch.tensor, alpha: torch.tensor,
                      wires: qml.wires.Wires, k: int, embed_func: callable,
                      transform_func: callable, diag_func: callable, observable: list,
                      embed_func_params: dict={}, transform_func_params: dict={}) -> float:
	"""
	:param x: The value in teach dimension of a data point of a series at time t.
	:param t: The time corresponding to the data point x.
	:param D: Diagonal entries of D.
	:param alpha: Parameters of W with the weights for each layer, each qubit, and each rotation
                      on individual dimensions in that order.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param embed_func_params: Optional additional parameters for embedding.
	:param transform_func_params: Optional additional parameters for rewinding.
	:return expval: The expected value of the circuit output.
	Simulate the circuit with a single data point and calculate the expected value.
	"""
	embed_func(x, wires=wires, **embed_func_params)
	transform_func(alpha, wires, **transform_func_params)
	diag_func(D*t, n_qubits, k=k)
	qml.adjoint(transform_func)(alpha, wires=range(n_qubits), **transform_func_params)
	coeffs = np.ones(len(wires))/len(wires)
	H = qml.Hamiltonian(coeffs, observable)
	expval = qml.expval(H)
	return expval

def get_fisher(x, t, alphas, eta_0, M_sample_func, sigmas, mus, N_E, wires, k, embed_func, transform_func, diag_func, observable, embed_func_params={}, transform_func_params={}): # do this for 10 and then do the distance measure :)
	threshold = 0.01439557393561154
	global x_g
	x_g = np.array([x])
	global t_g
	t_g = t
	global wires_g
	wires_g = wires
	global k_g
	k_g = k
	global embed_func_g
	embed_func_g = embed_func
	global transform_func_g
	transform_func_g = transform_func
	global diag_func_g
	diag_func_g = diag_func
	global observable_g
	observable_g = observable
	global embed_func_params_g
	embed_func_params_g = embed_func_params
	global transform_func_params_g
	transform_func_params_g = transform_func_params
	params = torch.cat((torch.flatten(alphas), torch.flatten(mus), torch.flatten(sigmas)))
	expval = [fisher(params) for i in range(10)]
	jacs = [torch.autograd.functional.jacobian(fisher, params) for i in range(10)]
	average_jacs = np.zeros(24)
	for j in range(10):
		for i in range(24):
			average_jacs[i] += jacs[j][i]
	average_jacs /= 10
	eta = params[23]
	u = ((eta-(sum(expval)/len(expval)))**2).detach().numpy()
	u_prime = 2*(eta-(sum(expval)/len(expval))).detach().numpy()
	comp_1 = (-1/threshold)/(((1/threshold)*u)+1)
	comp_2 = u_prime
	comp_3 = -1*average_jacs
	ders = comp_1*comp_2*comp_3
	current_fish = []
	global global_fish
	global fish_counter
	fish_counter += 1
	for der_1 in ders:
		row = []
		for der_2 in ders:
			row.append(der_1*der_2)
		current_fish.append(row)
	for i in range(len(global_fish)):
		for j in range(len(global_fish)):
			global_fish[i][j] += current_fish[i][j]

def sample_M(sigma: float, mus: float):
	"""
	:param sigma: The standard deviation of the normal distribution which is sampled from.
	:param mu: The mean of thenormal distribution which is sampled from.
	:return D: The sampled value.
	Sample values from a normal distribution.
	"""
	D = torch.normal(mus, sigma.abs())
	return D

def get_initial_parameters(transform_func, transform_func_layers, n_qubits,
					 num_distributions, init_flag) -> dict:
	"""
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param transform_func_layers: How many layers to use in the rewinding circuit.
	:param n_qubits: The number of qubits composing the circuit.
	:param num_distributions: The number of elements to sample in D.
	:return init_parameters: Randomly generated initial parameters.
	Generate initial parameters of the circuit to be trained.
	"""
	global initial_alphas_g
	global initial_mus_g
	global initial_sigmas_g
	global initial_eta_0_g
	if init_flag:
		alphas_shape = (transform_func_layers, n_qubits, 3) #might be with three as the first param idk
		initial_alphas = torch.tensor(np.random.uniform(0, 2*np.pi, size=alphas_shape), requires_grad=True).type(torch.DoubleTensor)
		initial_mus = torch.tensor(np.random.uniform(0, 2*np.pi, 3), requires_grad=True).type(torch.DoubleTensor)
		initial_sigmas = torch.tensor(np.random.uniform(0, 2*np.pi, 3), requires_grad=True).type(torch.DoubleTensor)
		initial_eta_0 = torch.tensor(np.random.uniform(-1, 1), requires_grad=True).type(torch.DoubleTensor)
		initial_alphas_g = initial_alphas
		initial_mus_g = initial_mus
		initial_sigmas_g = initial_sigmas
		initial_eta_0_g = initial_eta_0
		init_parameters = {'alphas': initial_alphas, 'mus': initial_mus, 'sigmas': initial_sigmas, 'eta_0': initial_eta_0}
	else:
		init_parameters = {'alphas': initial_alphas_g, 'mus': initial_mus_g, 'sigmas': initial_sigmas_g, 'eta_0': initial_eta_0_g}
	return init_parameters

def transform_func(alpha, wires, **transform_func_params):
	for layer in range(alpha.shape[0]):
		for qubit in range(alpha.shape[1]):
			qml.RX(alpha[layer][qubit][0], wires=qubit)
			qml.RY(alpha[layer][qubit][1], wires=qubit)
			qml.RZ(alpha[layer][qubit][2], wires=qubit)
		qml.CNOT(wires=[0, 1])

penalty = arctan_penalty
M_sample_func = sample_M
taus = torch.tensor([15])
embed_func = qml.templates.AngleEmbedding
transform_func = transform_func
diag_func = utils.create_diagonal_circuit
N_E = 10
observable = [qml.PauliZ(i) for i in range(n_qubits)]
x_batch_size = 10
t_batch_size = 10

noise = torch.from_numpy(utils.rescale(utils.load_data_from_file("data/noise.pkl")))
noise_train = noise[:5000]

for i in range(3320): #3340, 3340, 3320
	if i%100 == 0:i
		if i > 0:
			global_fish /= 100
			current_eigs = np.linalg.eig(global_fish)[0]
			eigs.append(current_eigs)
			global_fish = np.zeros((24, 24))
		init_flag = True
	else:
		init_flag = False
	params = get_initial_parameters(transform_func, 3, 2, 3, init_flag)
	alphas, mus, sigmas, eta_0 = params["alphas"], params["mus"], params["sigmas"], params["eta_0"]
	x_cycler = get_series_training_cycler(noise_train, 1)
	t_cycler = get_timepoint_training_cycler(noise_train, 1)
	t_indx = next(t_cycler)[0].numpy()
	t = np.linspace(0, 2*np.pi, 4096, endpoint=True)[t_indx]
	x = next(x_cycler)[0][0][t_indx].numpy()
	get_fisher(x, t, alphas, eta_0, M_sample_func, sigmas, mus, 10, range(2), 2, embed_func, transform_func, diag_func, observable)
eigs_path = "eigs.pkl"
pathlib.Path(os.path.abspath(eigs_path)).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(eigs, open(os.path.abspath(eigs_path), "wb"))
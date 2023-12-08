# This script is for running the experiments on an ibm_computer. It contains hard coded circuit parameters which are the final parameters of one of the training runs.

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import RYGate, RZGate, CXGate
import matplotlib.pyplot as plt
from itertools import combinations
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit_ibm_runtime.exceptions import RuntimeJobFailureError
from qiskit_ibm_runtime.exceptions import IBMRuntimeError
from qiskit.quantum_info import SparsePauliOp
import vars
import utils
import torch
import numpy as np
import sys
import pathlib
import pickle
import os
import pennylane as qml
from datetime import datetime
import time

def circ(x, gamma):
	# Embed
	embed = QuantumCircuit(2)
	embed.rx(x, 0)

	# W
	rot = QuantumCircuit(2)
	rot.rx(2.6164, 0)
	rot.ry(2.3002, 0)
	rot.rz(2.2215, 0)
	rot.rx(4.4286, 1)
	rot.ry(4.2454, 1)
	rot.rz(1.4864, 1)
	rot.cx(0, 1)

	rot.rx(2.9914, 0)
	rot.ry(3.1739, 0)
	rot.rz(2.2290, 0)
	rot.rx(2.0089, 1)
	rot.ry(4.9078, 1)
	rot.rz(1.4518, 1)
	rot.cx(0, 1)

	rot.rx(2.4362, 0)
	rot.ry(4.1635, 0)
	rot.rz(2.3998, 0)
	rot.rx(2.4489, 1)
	rot.ry(2.5148, 1)
	rot.rz(4.7999, 1)
	rot.cx(0, 1)
	rot = Operator(rot.to_gate())

	# D
	diag = QuantumCircuit(2)
	diag.rz(float(gamma[0]), 0)
	diag.rz(float(gamma[1]), 1)
	diag.cx(0, 1)
	diag.rz(float(gamma[2]), 1)
	diag.cx(0, 1)
	diag = Operator(diag.to_gate())

	# W_herm
	rot_t = rot.conjugate().transpose()

	# circ
	qc = QuantumCircuit(2)
	qc.append(embed, [0, 1])
	qc.append(rot, [0, 1])
	qc.append(diag, [0, 1])
	qc.append(rot_t, [0, 1])
	return qc

# generate circuits
noise = list(torch.from_numpy(utils.rescale(utils.load_data_from_file(vars.NONANOM_PATH)))[78:156])
events = list(torch.from_numpy(utils.rescale(utils.load_data_from_file(vars.ANOM_PATH))))

noise = noise[:1]
events = events[:1]

signals = [noise[0], events[0]]
signals = [events[0]]
times = np.linspace(0, 2*np.pi, 4096, endpoint=True)

circuits = []
for signal in signals:
	indxs = list(range(0, 4096, 59))
	signal = signal[0]
	signal = signal[indxs]
	times = times[indxs]
	for x_no in range(0, len(signal)):
		x = float(signal[x_no])
		time = times[x_no]
		for sample in range(10):
			gamma = torch.normal(torch.tensor([1.8834, 1.4833, 1.4896]), torch.tensor([5.0109, 3.8625, 1.5863]))
			circuit = circ(x, time*gamma)
			circuits.append(circuit)
pathlib.Path(os.path.abspath("circuits.pkl")).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(circuits, open(os.path.abspath("circuits.pkl"), "wb"))
print("circuits generated and saved")

observable = SparsePauliOp(["IZ", "ZI"], coeffs=[0.5, 0.5])
observables = []
for circuit_no in range(len(circuits)):
	observables.append(observable)
pathlib.Path(os.path.abspath("observables.pkl")).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(observables, open(os.path.abspath("observables.pkl"), "wb"))
print("observables generated and saved")

# submit to IBM
with open("circuits.pkl", "rb") as file:
	circuits = np.load(file, allow_pickle=True)
with open("observables.pkl", "rb") as file:
	observables = np.load(file, allow_pickle=True)
service = QiskitRuntimeService(channel="ibm_quantum", token="")  # Add the token here
backend = service.backend("ibm_kyoto")
with Session(service=service, backend=backend):
	estimator = Estimator()
	jobs = []
	circuits_per_signal = 700
	start_idx = 600
	while start_idx < circuits_per_signal:
		try:
			print("trying ", start_idx, " from ", circuits_per_signal)
			end_idx = start_idx+10
			jobs.append(estimator.run(circuits[start_idx:end_idx], observables[start_idx:end_idx], shots=256))
			start_idx = end_idx
		except IBMRuntimeError:
			start_teim = datetime.strptime("30/11/23 11:00:00", "%d/%m/%y %H:%M:%S")
			check_jobs = service.jobs(limit=9999, created_after=start_teim)
			print("failed, retrying ", start_idx, " to ", end_idx)
			flag = True
			time_counter = 0
			while flag:
				flag = False
				time.sleep(120)
				time_counter += 2
				if time_counter%2 == 0:
					print("waited for ", time_counter, "minutes")
				for check_job in check_jobs:
					if check_job.status().value == "job is queued":
						flag = True
print("noise test sumbitted")


# analyse results
eta = -0.0311
service = QiskitRuntimeService(channel="ibm_quantum", token="d205ea309f8b587b8d630b1a6c2df67bd9dfe771606b4a7cae15399fc883feb80ae1c6f115c0559e5a5540a8d2ae73d011843a00cb5cd75bbc74ee2f2ac03be2")
backend = service.backend("ibm_kyoto")
job_means = []
time = 0
counter = 0
segment_start = datetime.strptime("", "%d/%m/%y %H:%M:%S")  # Add form when results should be retrieved.
segment_end = datetime.strptime("", "%d/%m/%y %H:%M:%S")  # Add until when results should be retrieved.
batches = service.jobs(limit=9999, created_after=segment_start, created_before=segment_end)

jobs = []
for batch in batches:
	if batch.status().value == "job has successfully run":
		jobs.append(batch)
job_no = 0
for job in jobs:
	try:
		if job.result() is not None:
			job_no += 1
			job_values = job.result().values
			job_values_list = []
			for value in job_values:
				job_values_list.append(value)
			job_means.append(sum(job_values_list)/len(job_values_list))
		else:
			print("missing job contribution")
		time += job.metrics()["usage"]["quantum_seconds"]
		print("job ", job_no, " of ", len(jobs))
		print(len(job_means), " job means")
		print()
	except RuntimeJobFailureError:
		print("missing job and time contributions")
shifted_results = []
for i in job_means:
	shifted_results.append((eta-i)**2)
print(sum(shifted_results)/len(shifted_results))
print(shifted_results)
print(time)


# check status
"""
service = QiskitRuntimeService(channel="ibm_quantum", token="")  # Add a token here.
backend = service.backend("ibm_kyoto")
segment_start = datetime.strptime("", "%d/%m/%y %H:%M:%S")  # Add form when results should be retrieved.
segment_end = datetime.strptime("", "%d/%m/%y %H:%M:%S")  # Add until when results should be retrieved.
jobs = service.jobs(limit=9999, created_after=start_date)
for job in jobs:
	print(job.status().value)
	print()
"""
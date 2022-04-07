import dimod
import neal
import matplotlib.pyplot as plt
import numpy as np
import argparse
from math import*
from Tools import*
from Network import*
from random import*
from tqdm import tqdm
import os

import torch

from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation with D-Wave support')
parser.add_argument(
    '--dataset',
    type=str,
    default='2d_class',
    help='Dataset we use for training (default=2d_class, others: digits, mnist)')
parser.add_argument(
    '--mode',
    type=str,
    default='ising',
    help='Which problem we submit to the QPU (default=qubo (0/1 spins), others: ising (-1/+1 variables: à priori no rescaling après))')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 10, 10],
    help='List of layer sizes (default: 10)')
parser.add_argument(
    '--n_iter_free',
    type=int,
    default=1,
    help='Times to iterate for the QPU on a single data point for getting the minimal energy state of the free phase (default=1)')
parser.add_argument(
    '--n_iter_nudge',
    type=int,
    default=1,
    help='Times to iterate for the QPU on a single data point for getting the minimal energy state of the nudge phase (default=5)')
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=1,
    help='fraction of system non-annealed (default=0.5, if <0.5: more annealing, if > 0.5, less annealing)')
parser.add_argument(
    '--N_data',
    type=int,
    default=10,
    help='Number of data points for training (default=100)')
parser.add_argument(
    '--N_data_test',
    type=int,
    default=10,
    help='Number of data points for testing (default=10)')
parser.add_argument(
    '--beta',
    type=float,
    default=1,
    help='Beta - hyperparameter of EP (default=10)')
parser.add_argument(
    '--random_beta',
    type=int,
    default=0,
    help='Random sign of beta or not - hyperparameter of EP (default=0, other: 1)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Size of mini-batches we use (for training only)')
parser.add_argument(
    '--lrW0',
    type=float,
    default=0.0001,
    help='Learning rate for weights - input-hidden  (default=0.01)')
parser.add_argument(
    '--lrW1',
    type=float,
    default=0.0001,
    help='Learning rate for weights - hidden-output (default=0.01)')
parser.add_argument(
    '--lrB0',
    type=float,
    default=0.0001,
    help='Learning rate for biases - hidden (default=0.001)')
parser.add_argument(
    '--lrB1',
    type=float,
    default=0.0001,
    help='Learning rate for biases - output (default=0.001)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='Number of epochs (default=10)')
parser.add_argument(
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model to continue the training (default=0, else = 1)')
parser.add_argument(
    '--init',
    type=str,
    default='kaiming_uniform',
    help='Type of initialization to use (default=random_normal, others: random_uniform, kaiming_normal, kaiming_uniform, glorot_normal, glorot_uniform)')
parser.add_argument(
    '--gain_weight0',
    type=float,
    default=0.1,
    help='Gain for initialization of the weights - input-hidden (default=1)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.1,
    help='Gain for initialization of the weights  - hidden-output (default=1)')
parser.add_argument(
    '--gain_bias0',
    type=float,
    default=0.0,
    help='Gain for initialization of the biases - hidden layer (default=1)')
parser.add_argument(
    '--gain_bias1',
    type=float,
    default=0.0,
    help='Gain for initialization of the biases - output layer (default=1)')
parser.add_argument(
    '--bias_lim',
    type=float,
    default=4.0,
    help='Max limit for the amplitude of the local biases applied to the qbits, either for free or nudge phase - (default=1)')
parser.add_argument(
    '--chain_strength',
    type=float,
    default=1.0,
    help='Value of the coupling in the chain of identical qbits - (default=4)')
parser.add_argument(
    '--auto_scale',
    type=int,
    default=0,
    help='Set auto_scale or not for the problems - (default=False)')
parser.add_argument(
    '--epoch_decay',
    type=float,
    default=0,
    help='Every epoch_decay, we decay the learning rate by the quantity defined bellow (default=10)')
parser.add_argument(
    '--decay_lr',
    type=float,
    default=1,
    help='By how much we decay the learning rate every X epochs (default=2)')
args = parser.parse_args()

if args.auto_scale == 0:
    args.auto_scale = False
else:
    args.auto_scale = True

with torch.no_grad():
    qpu_sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}, auto_scale = False))

    BASE_PATH = createPath(args)
    dataframe = initDataframe(BASE_PATH)

    if args.dataset == "2d_class":
        train_loader, test_loader = generate_data(args)
    elif args.dataset == "digits":
        train_loader, test_loader = generate_digits(args)
    elif args.dataset == "mnist":
        train_loader, test_loader, dataset = generate_mnist(args)

    if args.load_model == 0:
        saveHyperparameters(BASE_PATH, args)
        net = Network(args)
    else:
        net = load_model_numpy(BASE_PATH)

    exact_loss_tab, exact_falsePred_tab = [], []
    exact_loss_test_tab, exact_falsePred_test_tab = [], []

    qpu_loss_tab, qpu_falsePred_tab = [], []
    qpu_loss_test_tab, qpu_falsePred_test_tab = [], []

    plot_functions(net, BASE_PATH, prefix, -1, args)

    for epoch in tqdm(range(args.epochs)):
        net.confusion_matrix_train = np.zeros((10,10))
        net.confusion_matrix_test = np.zeros((10,10))

        exact_loss, exact_falsePred, qpu_loss, qpu_falsePred = train(net, args, train_loader, simu_sampler, exact_sampler, qpu_sampler)
        exact_loss_tab.append(exact_loss)
        exact_falsePred_tab.append(exact_falsePred)
        qpu_loss_tab.append(qpu_loss)
        qpu_falsePred_tab.append(qpu_falsePred)

        exact_loss, exact_falsePred, qpu_loss, qpu_falsePred = test(net, args, test_loader, simu_sampler, exact_sampler, qpu_sampler)
        exact_loss_test_tab.append(exact_loss)
        exact_falsePred_test_tab.append(exact_falsePred)
        qpu_loss_test_tab.append(qpu_loss)
        qpu_falsePred_test_tab.append(qpu_falsePred)

        dataframe = updateDataframe(BASE_PATH, dataframe, np.array(exact_falsePred_tab)[-1]/len(train_loader.dataset)*100, np.array(exact_falsePred_test_tab)[-1]/len(test_loader.dataset)*100, np.array(qpu_falsePred_tab)[-1]/len(train_loader.dataset)*100, np.array(qpu_falsePred_test_tab)[-1]/len(test_loader.dataset)*100, exact_loss_tab, exact_loss_test_tab, qpu_loss_tab, qpu_loss_test_tab)

        save_model_numpy(BASE_PATH, net)

        plot_functions(net, BASE_PATH, prefix, epoch, args)
        if (epoch % args.epoch_decay == 0) and epoch != 0:
            args.lrW0 = args.lrW0/ args.decay_lr
            args.lrW1 = args.lrW1/ args.decay_lr
            args.lrB0 = args.lrB0/ args.decay_lr
            args.lrB1 = args.lrB1/ args.decay_lr
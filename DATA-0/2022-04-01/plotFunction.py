import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def plot_error(dataframe, idx):

    ave_train_error_tab = np.array(dataframe['Exact_Train_Error'].values.tolist())
    ave_test_error_tab = np.array(dataframe['Exact_Test_Error'].values.tolist())

    single_train_error_tab = np.array(dataframe['QPU_Train_Error'].values.tolist())
    single_test_error_tab = np.array(dataframe['QPU_Test_Error'].values.tolist())

    # plt.figure()
    # plt.plot(ave_train_error_tab, label = 'Exact train error #' + str(idx))
    # plt.plot(ave_test_error_tab, label = 'Exact test error #' + str(idx))

    plt.plot(single_train_error_tab, '--', label = 'QPU train accuracy #' + str(idx))
    plt.plot(single_test_error_tab, '--', label = 'QPU test accuracy #' + str(idx))

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test Accuracy (averaged)')
    plt.legend()

    return ave_train_error_tab, ave_test_error_tab, single_train_error_tab, single_test_error_tab


def plot_loss(dataframe, idx):

    exact_train_loss_tab = np.array(dataframe['Exact_Train_Loss'].values.tolist())
    exact_test_loss_tab = np.array(dataframe['Exact_Test_Loss'].values.tolist())

    qpu_train_loss_tab = np.array(dataframe['QPU_Train_Loss'].values.tolist())
    qpu_test_loss_tab = np.array(dataframe['QPU_Test_Loss'].values.tolist())


    #plt.plot(exact_train_loss_tab, label = 'Exact train loss #' + str(idx))
    #plt.plot(exact_test_loss_tab, label = 'Exact test loss #' + str(idx))

    plt.plot(qpu_train_loss_tab, '--', label = 'QPU train loss #' + str(idx))
    plt.plot(qpu_test_loss_tab, '--', label = 'QPU test loss #' + str(idx))

    plt.ylabel('Loss (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test loss')
    plt.legend()

    return qpu_train_loss_tab, qpu_test_loss_tab


def plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error):
    '''
    Plot mean train & test error with +/- std
    '''
    try:
        store_ave_train_error, store_ave_test_error = np.array(store_ave_train_error), np.array(store_ave_test_error)
        mean_ave_train, mean_ave_test = np.mean(store_ave_train_error, axis = 0), np.mean(store_ave_test_error, axis = 0)
        std_ave_train, std_ave_test = np.std(store_ave_train_error, axis = 0), np.std(store_ave_test_error, axis = 0)

        store_single_train_error, store_single_test_error = np.array(store_single_train_error), np.array(store_single_test_error)
        mean_single_train, mean_single_test = np.mean(store_single_train_error, axis = 0), np.mean(store_single_test_error, axis = 0)
        std_single_train, std_single_test = np.std(store_single_train_error, axis = 0), np.std(store_single_test_error, axis = 0)

        epochs = np.arange(0, len(store_ave_test_error[0]))

        plt.figure()
        plt.plot(epochs, mean_ave_train, label = 'mean_ave_train_accuracy')
        plt.fill_between(epochs, mean_ave_train - std_ave_train, mean_ave_train + std_ave_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_ave_test, label = 'mean_ave_test_accuracy')
        plt.fill_between(epochs, mean_ave_test - std_ave_test, mean_ave_test + std_ave_test, facecolor = '#fadcb3')

        plt.plot(epochs, mean_single_train, '--', label = 'mean_single_train_accuracy')
        plt.fill_between(epochs, mean_single_train - std_single_train, mean_single_train + std_single_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_single_test, '--', label = 'mean_single_test_accuracy')
        plt.fill_between(epochs, mean_single_test - std_single_test, mean_single_test + std_single_test, facecolor = '#fadcb3')

        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epochs')
        plt.title('Mean train and Test Accuracy with std')
        plt.legend()

    except:
        pass

    return 0


def plot_mean_loss(store_train_loss, store_test_loss):
    '''
    Plot mean train & test loss with +/- std
    '''
    try:
        store_train_loss, store_test_loss = np.array(store_train_loss), np.array(store_test_loss)
        mean_train, mean_test = np.mean(store_train_loss, axis = 0), np.mean(store_test_loss, axis = 0)
        std_train, std_test = np.std(store_train_loss, axis = 0), np.std(store_test_loss, axis = 0)
        epochs = np.arange(0, len(store_test_loss[0]))
        plt.figure()
        plt.plot(epochs, mean_train, label = 'mean_train_loss')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_test, label = 'mean_test_loss')
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, facecolor = '#fadcb3')

        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Mean train and Test loss with std')
        plt.legend()

    except:
        pass

    return 0

if __name__ == '__main__':
    if os.name != 'posix':
        path = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        path = os.getcwd()
        prefix = '/'

    files = glob.glob('*')
    store_ave_train_error, store_single_train_error, store_train_loss = [], [], []
    store_ave_test_error, store_single_test_error, store_test_loss = [], [], []

    plt.figure()
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
            #plot error
            ave_train_error_tab, ave_test_error_tab, single_train_error_tab, single_test_error_tab = plot_error(DATAFRAME, name)
            store_ave_train_error.append(ave_train_error_tab)
            store_ave_test_error.append(ave_test_error_tab)
            store_single_train_error.append(single_train_error_tab)
            store_single_test_error.append(single_test_error_tab)

        else:
            pass

    plt.figure()
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
            #plot loss
            train_loss_tab, test_loss_tab = plot_loss(DATAFRAME, name)
            store_train_loss.append(train_loss_tab)
            store_test_loss.append(test_loss_tab)
        else:
            pass

    plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error)
    # plot_mean_loss(store_train_loss, store_test_loss)


    plt.show()
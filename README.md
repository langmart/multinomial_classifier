A multilayer perceptron framework for the classification of events into multiple
mutually exclusive categories. It is derived from an older version of a previous 
multilayer perceptron framework for binary classification, developed by Max Welsch
(https://github.com/welschma/NNFlow).

#Usage: 

##Preprocessing the data

The loaded data is available in a structured-array format and has to be
converted into numpy arrays with the preprocessing script. This script is
invoked with the multi_get_branches.py file, where the following can be
specified:


* The input paths for the training (Even) and validation (Odd) sets, each
    consisting of signal (ttH) and background (ttbarSL = ttbar semi-leptonic) 
    events. 

*   The branchlists to be converted; expects a list of strings, each of them
    describing a file containing the branches to be converted.

*   The categories to be converted; expects a list os list of strings. Each
    of the lists will be extracted. The code for the background events is: 
        
    *   '30':   tt + bb
    *   '20':   tt + 2b
    *   '10':   tt + b
    *   '01':   tt + cc
    *   '00':   tt + light flavor 

    Signal events will always be converted.

*   preselection: 
            
    *   'no':       no preselection
    *   'weak':     >= 6 jets, >=3 btags (medium)
    *   'strong':   >= 6 jets, >=4 btags (medium)

*   The output directory.


The preprocessing utility also applies weights to the events and generates
one-hot labels with dimension 1 + num(categories).




##Training a neural network

The training of a neural network is invoked using the train.py or
train_ttH.py file. Several parameters are to be set prior to launch:

*   The paths to the training and validation sets. A path to a file 
    containg the cross section weights for signal and background is
    needed in order to plot a weighted heatmap. In order to determine the
    most heavily weighted input features, a path to the converted branchlist
    must be given.

*   outpath and exec_name:  The models will be stored in concat(outpath,
    exec_name). 

*   Labels: The labels of the converted categories. Only needed for plotting
    reasons.

*   normalization: The normalization algorithm to use. Can be either
    'gaussian' or 'minmax'.

*   outsize: The dimension of the output vector. Must be the same as
    len(labels).

*   optname: The optimizer to choose. The following are implemented:
        
    *   'Adam':         Adam optimizer
    *   'GradDescent':  Gradient Descent optimizer
    *   'Adagrad':      Adagrad optimizer
    *   'Adadelta':     Adadelta optimizer
    *   'Momentum':     Momentum optimizer

*   optimizer_options: The options for the chosen optimizer. Refer to the 
    train.py and train_ttH.py files and to the perceptron framework for
    details of possible values. If given an empty list, the default options
    are used.

*   act_func: The activation function. The following values are supported: 

    *   'tanh'
    *   'relu'
    *   'elu'
    *   'softplus'
    *   'sigmoid

    If no activation function is chosen, or another value is passed, 'tanh'
    will be chosen.

*   beta: The L2 regularization parameter.

*   N_EPOCHS: Number of epochs to be trained if no early stopping occurs.

*   learning_rate: Optimizer learning rate.

*   batch_size: Size of the training batch.

*   hidden_layers: A list containing the number of neurons in each layer.
    The list [200, 100, 100] will produce a neural network with three hidden
    layers, of which the first contains 200 neurons, while the latter two
    letters each contain 100 neurons.

*   keep_prob: The firing probability for a neuron. Must be a value 0 <
    keep_prob <= 1.

*   decay_learning_rate: Indicates whether to manually decay the learning
    rate. Allowed values are 'yes' and 'no'.

*   lrate_decay_options: A list containing the decay rate and the step
    width, e.g. [0.97, 200]

*   batch_decay: Indicates whether to change the batch size during the
    training process. Allowed values are 'yes' and 'no'.

*   batch_decay_options: Sets the decay rate and step width for the batch
    size decay. The step_width is given in number of epochs.

*   ttH_penalty: A penalty value for false positives and false negatives
    when using the ttH classifier. Not usable with train.py.


With these options, the neural network is built and trained. During the
training process, plots are generated and stored. 

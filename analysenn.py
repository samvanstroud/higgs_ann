import nn
import time
import numpy as np
import math
import csv
from random import shuffle
import myreadlhe


def train_test_set(data_set, n):
    ''' Creates a shuffled list of n labelled training data examples to 
    train the neural network on. 
    Make n here the fraction of the data_set that is kept by multiplying 
    by it's length '''

    # shuffle the set
    shuffle(data_set)

    # keep the first n items
    training_set = data_set[:n]
    #training_set_labels = [event.pop('y', None) for event in training_set]

    testing_set = data_set[n:]
    #training_set_labels = [event.pop('y', None) for event in testing_set]

    return training_set, testing_set

def normalize_inputs(list_of_inputs):
    '''Normalizes a list of dictionaries by subtracting the mean
    for each data point and dividing the result by the stand dev'''
    
    input_stats = {}

    # for each key present in the dictionaries
    for key in list(list_of_inputs[0]):

        # don't do stats for the output labels
        if key != 'y':
            # get the list of values for that key
            values = np.array([d[key] for d in list_of_inputs])

            # find statistical values
            mean = np.mean(values)
            sigma = 1 if np.std(values) == 0 else np.std(values)
            
            # add values to a dictionary for each input 
            input_stats.update({key: [mean, sigma]})

    # set mean and stdev for the labels such that they are unchanged
    # hacky way to not change labels y
    input_stats['y'] = [0,1]

    normalized_inputs = [{key:(value - input_stats[key][0])/input_stats[key][1] 
                        for key, value in event.items()}
                        for event in list_of_inputs]

    # return the normalised list of inputs
    return normalized_inputs



def init_data_set(sig_path, bkg_path):
    ''' Read data from a .lhe file and convert the data into a list of
    events represented by python dictionaries '''

    print("Reading data...")

    # initialize the object
    data_set = np.array([])

    # read signal file
    sig_data = myreadlhe.read_event_file(sig_path)

    # label signal
    for unlabelled_input in sig_data:
        unlabelled_input['y'] = 1

    # add labelled signal to the data set
    data_set = np.append(data_set, sig_data)

    # read background file
    bkg_data = myreadlhe.read_event_file(bkg_path)
    
    # label background
    for unlabelled_input in bkg_data:
        unlabelled_input['y'] = 0

    # add lablled background to the training set
    data_set = np.append(data_set, bkg_data)

    # normalise data
    print("Normalising training set...")
    data_set = normalize_inputs(data_set)

    return data_set


def get_result(network, testing_set, n, high, low):
    ''' A function which takes a trained network and a evaluation set
    and returns a tuple of the false and true positive rates for 
    some cutoff value. 
    NOTE: there is no uncertain area where we don't classify. Hence
    only one cutoff bound is used and the 'low' argument is redundant. 
    Look into the effectiveness of changing this.
    '''

    # initialise variables
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # for each test event
    for test in testing_set[:n]:

        # get the expect result
        expected = test['y']

        # get the hypothesis from the ANN
        result = nn.feed_forward(network, test)[-1][0]

        # categorise the result
        if expected == 1 and result > high:
            true_positives += 1
        elif expected == 1 and result < high:
            false_negatives += 1
        elif expected == 0 and result > high:
            false_positives += 1
        elif expected == 0 and result < high:
            true_negatives += 1

    # calculate and return true and false positive rates
    tp_rate = true_positives / (true_positives + false_negatives)
    fp_rate = false_positives / (false_positives + true_negatives)

    return fp_rate, tp_rate


def get_histogram(network, training_set, testing_set, n):
    num_of_training_examples = len(training_set)

    # separate out the signal and background events in the training set
    #signal_events = [event for event in training_set if event['y'] == 1]
    #background_events = [event for event in training_set if event['y'] == 0]

    # get the classifier response to signal and background events separately
    training_sig = [nn.feed_forward(network, event)[-1][0] for event in 
                    [sig for sig in training_set if sig['y'] == 1]]
    training_bkg = [nn.feed_forward(network, event)[-1][0] for event in 
                    [bkg for bkg in training_set if bkg['y'] == 0]]

    # Get results of testing data
    test = [nn.feed_forward(network, event)[-1][0] for event in testing_set[:n]]

    # Get histograms for the training data
    training_sig_hist = np.histogram(training_sig, bins=50, range=(0.0,1.0))[0] / num_of_training_examples
    training_bkg_hist = np.histogram(training_bkg, bins=50, range=(0.0,1.0))[0] / num_of_training_examples

    # Get histogram for the testing data
    test_hist = np.histogram(test, bins=50, range=(0.0,1.0))[0] / n

    # Normalize the testing data histogram
    #test_hist = test_hist / n
    
    # Get the x coordaintes for each plotted point
    x_vals = np.arange(0,1.02,0.02)

    #data = zip(x_vals, training_sig_hist, np.add(training_bkg_hist,training_sig_hist) , test_hist)
    data = zip(x_vals, training_sig_hist, training_bkg_hist , test_hist)

    fname = str(num_of_training_examples)

    with open('saved_data\\hist\\' + fname + '.csv','w',newline="\n", encoding="utf-8") as out:
        # create the csv writer
        csv_out=csv.writer(out)

        # write the heading row to the file
        csv_out.writerow(['a','b','c','d'])

        # write each x,y coordainte to the file
        for row in data:
            csv_out.writerow(row)
  


def get_roc(network, testing_set, n):
    ''' A function which generates the data needed to plot a
    ROC curve by calculating a variety signal cutoff values.'''

    # set the resolution of the ROC curve
    step = 0.025

    # get the x,y coordinates for each cutoff value
    roc_data = [get_result(network, testing_set, n, cutoff, cutoff) 
                    for cutoff in np.arange(0,1 + step,step)]
    #print(roc_data)
    unzipped = list(zip(*roc_data))

    # calculate the area under the ROC curve
    area = abs(np.trapz(unzipped[1], unzipped[0]))

    return area, roc_data



def write_roc_csv(data, area, fname):
    ''' A function that takes a list of 2d-tuples and writes each
    value to a new row in a CSV file. The headings are labelled 
    'a' and 'b' to allow for easy plotting with LaTeX's pgfplots 
    package '''
    with open('saved_data\\roc\\' + fname + '.csv','w',newline="\n", encoding="utf-8") as out:
        # create the csv writer
        csv_out=csv.writer(out)

        csv_out.writerow(["area" , str(area)])

        # write the heading row to the file
        csv_out.writerow(['a','b'])

        # write each x,y coordainte to the file
        for row in data:
            csv_out.writerow(row)


def train_n(n):
    ''' Trains n networks of the same configuration for a wider search of the 
    solution space. The best result is saved to a CSV file.'''
    
    # Setup
    best_score = 0
    best_data = []
    best_network = []
    total_area = 0
    best_training_set = []
    best_testing_set = [] 

    # Read in the event data
    data_set = init_data_set(mu_e_sig_path, mu_e_bkg_path)

    for i in np.arange(1, n + 1):
        print("Training network " + str(i) + " of " + str(n) + "...")

        # Initialize a new network
        this_network = nn.initialize_network([16, 16, 1])

        # Get a random training and testing set of a given size
        training_set, testing_set = train_test_set(data_set, TRAIN_SET_SIZE)

        # Train the network
        start_time = time.time()
        nn.train(this_network, training_set, testing_set, TRAIN_SET_IT, a=AFLAG)
        train_time = round(time.time() - start_time, 2)

        # Evaluate the network's performance
        start_time = time.time()
        roc_auc, data = get_roc(this_network, testing_set, TEST_COUNT)
        eval_time = round(time.time() - start_time, 2)

        print("The network was trained in " + str(train_time) + " s, " + 
              "and was evaluated in " + str(eval_time) + " s")

        # Keep count of the total score in order to find the average
        total_area += roc_auc

        # update values if we find a new best performing network
        if roc_auc > best_score:
            best_score = roc_auc
            best_data = data
            best_network = this_network
            best_training_set = training_set
            best_testing_set = testing_set

        print("This network scored " + str(round(roc_auc, 3)) + '. '
              "The best current score is " + str(round(best_score, 3)))

    # calculate the average score for the n networks
    average_score = round(total_area/n, 3)
    print("Average score over " + str(n) + " networks is " + str(average_score))

    # write the best ROC curve data to csv
    write_roc_csv(best_data, best_score, str(TRAIN_SET_SIZE))

    # write the best histogram to csv
    get_histogram(best_network, best_training_set, best_testing_set, TEST_COUNT)




# file paths: todo implement command line args
mu_e_sig_path = ".\\res\\pp_h_2mu2e_heft\\unweighted_events.lhe"
mu_e_bkg_path = ".\\res\\pp_2mu2e_bkg\\unweighted_events.lhe"
mu_nu_sig_path = ".\\res\\pp_h_2munu_heft\\unweighted_events.lhe"
mu_nu_bkg_path = ".\\res\\pp_2mu2nu_bkg\\unweighted_events.lhe"


# Add learning rate here
TRAIN_SET_SIZE = 100
TRAIN_SET_IT = 20
TEST_COUNT = 100
NUM_INIT = 3
AFLAG = False

# train some networks!
start1 = time.time()
train_n(NUM_INIT)
end1 = time.time() - start1
print(round(end1/NUM_INIT,2))



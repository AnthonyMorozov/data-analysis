#!/usr/bin/python

# Attribute Information:
# STG (The degree of study time for goal object materails),
# SCG (The degree of repetition number of user for goal object materails)
# STR (The degree of study time of user for related objects with goal object)
# LPR (The exam performance of user for related objects with goal object)
# PEG (The exam performance of user for goal objects)
# UNS (The knowledge level of user)

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import random
import math

class SOM:
    class Neuron:
        name = ''
        weights = []

        def __getitem__(self, key):
            return self.weights[key]

        def make_weights(self, weights_count):
            self.weights = []
            for idx in range(weights_count):
                self.weights.append(random.uniform(-1, 1))

        def set_name(self, name):
            self.name = name

    neurons = []
    learn_speed = 0.0

    def _make_neurons(self, clusters_count):
        self.neurons = []
        for idx in range(clusters_count):
            self.neurons.append(SOM.Neuron())

    def __init__(self, clusters_count, *learn_speed):
        self._make_neurons(clusters_count)
        self.set_learn_speed(*learn_speed)

    def reset(self, clusters_count):
        self._make_neurons(clusters_count)

    def set_learn_speed(self, learn_speed):
        self.learn_speed = learn_speed

    def _init_neurons(self, weight_count):
        for neuron in self.neurons:
            neuron.make_weights(weight_count)

    def _get_winner(self, data_entry):
        distances = []
        for neuron in self.neurons:
            distance = 0
            for idx in range(len(data_entry)):
                distance += (neuron[idx] - data_entry[idx]) ** 2
            distances.append(distance)
        return distances.index(min(distances))

    def fit(self, data, *clusters_names):
        train_data = list(data)
        time = 0
        speed = float(self.learn_speed)
        # Initialize all neurons with random weights
        self._init_neurons(len(train_data[0]))

        while len(train_data) != 0:
            time += 0.1
            entry_idx = random.randint(0, len(train_data) - 1)

            # Get the closest neuron in the list (winner)
            winner_idx = self._get_winner(train_data[entry_idx])

            # Calculating all changes in weight matrix
            sigma = 1 / math.exp(time ** -2)
            dw_vect = []
            for neuron in self.neurons:
                distance = 0
                for weight in enumerate(neuron.weights):
                    distance += (weight[1] - self.neurons[winner_idx][weight[0]]) ** 2
                # Gaussian function
                h = math.exp(-1 * (distance / sigma))
                difference = []
                for idx in range(len(train_data[entry_idx])):
                    difference.append(neuron.weights[idx] - train_data[entry_idx][idx])
                    difference[idx] = difference[idx] * h * speed
                dw_vect.append(difference)

            if speed > 0:
                speed -= 0.001

            # Making changes
            for neuron in enumerate(self.neurons):
                for idx in range(len(neuron[1].weights)):
                    neuron[1].weights[idx] -= dw_vect[neuron[0]][idx]

            del train_data[entry_idx]
            self.print_neurons()

    def print_neurons(self):
        for neuron in self.neurons:
            print '%3d ' % self.neurons.index(neuron),
            for weight in neuron.weights:
                print ' %4.10f ' % weight,
            print

    def predict(self, data):
        winners = []
        for data_entry in data:
            winners.append(self._get_winner(data_entry))
        return winners

#--------------------------------------------------------------------

uns_dict = {
    'very_low' : 1,
    'Low' : 2,
    'Middle' : 3,
    'High' : 4
}

index_dict = {
    'STG' : 0,
    'SCG' : 1,
    'STR' : 2,
    'LPR' : 3,
    'PEG' : 4,
    'UNS' : 5
}

def load_data(path):
    data = []
    for line in open(path):
        line_data = line.split("\t")
        line_data[0] = float(line_data[0])
        line_data[1] = float(line_data[1])
        line_data[2] = float(line_data[2])
        line_data[3] = float(line_data[3])
        line_data[4] = float(line_data[4])
        line_data[5] = str(line_data[5][:len(line_data[5]) - 1])
        line_data[5] = uns_dict[line_data[5]]
        data.append(line_data)
    return data

def draw_data_2d(data):
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    for line in data:
        x0.append(line[0])
        x1.append(line[1])
        x2.append(line[2])
        x3.append(line[3])
        x4.append(line[4])
        x5.append(line[5])
    fig = plt.figure()
    subfig1 = fig.add_subplot(221)
    subfig1.scatter(x1, x4)
    subfig2 = fig.add_subplot(222)
    subfig2.scatter(x2, x4)
    subfig3 = fig.add_subplot(223)
    subfig3.scatter(x1, x2)
    plt.show()

def draw_data_3d(data):
    x1 = []
    x2 = []
    x4 = []
    for line in data:
        x1.append(line[1])
        x2.append(line[2])
        x4.append(line[4])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, x4)
    ax.set_xlabel('SCG')
    ax.set_ylabel('STR')
    ax.set_zlabel('PEG')
    plt.show()

def print_data(data):
    for line in data:
        print line

#------------------------------------------------------------------
"""
data = load_data("./Train.txt")
print_data(data)
draw_data_2d(data)
draw_data_3d(data)

# Creating and training SOM
som = SOM(3, 0.25)
som.fit(data)

# Creating list with predictions
predictions = som.predict(data)

# Showing 2D plots with clusters on it
fig = plt.figure()
subfig1 = fig.add_subplot(221)
subfig2 = fig.add_subplot(222)
subfig3 = fig.add_subplot(223)
for line in enumerate(data):
    x1 = line[1][1]
    x2 = line[1][2]
    x4 = line[1][4]
    if predictions[line[0]] == 0:
        subfig1.scatter(x1, x4, c = 'r')
        subfig2.scatter(x2, x4, c = 'r')
        subfig3.scatter(x1, x2, c = 'r')
    elif predictions[line[0]] == 1:
        subfig1.scatter(x1, x4, c = 'g')
        subfig2.scatter(x2, x4, c = 'g')
        subfig3.scatter(x1, x2, c = 'g')
    elif predictions[line[0]] == 2:
        subfig1.scatter(x1, x4, c = 'b')
        subfig2.scatter(x2, x4, c = 'b')
        subfig3.scatter(x1, x2, c = 'b')
    elif predictions[line[0]] == 3:
        subfig1.scatter(x1, x4, c = 'y')
        subfig2.scatter(x2, x4, c = 'y')
        subfig3.scatter(x1, x2, c = 'y')
plt.show()

# Showing 3D scatterplot with clusters on it
x1 = []
x2 = []
x4 = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for line in data:
    x1 = line[1]
    x2 = line[2]
    x4 = line[4]
    prediction = int(som.predict([line])[0])
    if 0 == prediction:
        ax.scatter(x1, x2, x4, c = 'r', marker = 'o')
    elif 1 == prediction:
        ax.scatter(x1, x2, x4, c = 'g', marker = 'o')
    elif 2 == prediction:
        ax.scatter(x1, x2, x4, c = 'b', marker = 'o')
    elif 3 == prediction:
        ax.scatter(x1, x2, x4, c = 'y', marker = 'o')

plt.show()

# Alright, let's make test on the only 3 parameters
new_data = []
for line in data:
    new_data.append([line[1], line[2], line[4]])

new_som = SOM(3, 0.25)
new_som.fit(new_data)
new_predictions = new_som.predict(new_data)

# Showing 2D plots with clusters on it
fig = plt.figure()
subfig1 = fig.add_subplot(221)
subfig2 = fig.add_subplot(222)
subfig3 = fig.add_subplot(223)
for line in enumerate(new_data):
    x1 = line[1][0]
    x2 = line[1][1]
    x4 = line[1][2]
    if new_predictions[line[0]] == 0:
        subfig1.scatter(x1, x4, c = 'r')
        subfig2.scatter(x2, x4, c = 'r')
        subfig3.scatter(x1, x2, c = 'r')
    elif new_predictions[line[0]] == 1:
        subfig1.scatter(x1, x4, c = 'g')
        subfig2.scatter(x2, x4, c = 'g')
        subfig3.scatter(x1, x2, c = 'g')
    elif new_predictions[line[0]] == 2:
        subfig1.scatter(x1, x4, c = 'b')
        subfig2.scatter(x2, x4, c = 'b')
        subfig3.scatter(x1, x2, c = 'b')
    elif new_predictions[line[0]] == 3:
        subfig1.scatter(x1, x4, c = 'y')
        subfig2.scatter(x2, x4, c = 'y')
        subfig3.scatter(x1, x2, c = 'y')
plt.show()

# Showing 3D scatterplot with clusters on it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for line in enumerate(new_data):
    x1 = line[1][0]
    x2 = line[1][1]
    x4 = line[1][2]
    if 0 == new_predictions[line[0]]:
        ax.scatter(x1, x2, x4, c = 'r', marker = 'o')
    elif 1 == new_predictions[line[0]]:
        ax.scatter(x1, x2, x4, c = 'g', marker = 'o')
    elif 2 == new_predictions[line[0]]:
        ax.scatter(x1, x2, x4, c = 'b', marker = 'o')
    elif 3 == new_predictions[line[0]]:
        ax.scatter(x1, x2, x4, c = 'y', marker = 'o')
plt.show()

# Alright, trying more obvious clusters
# Let's make a cube, shall we?
data_lst = []
for idx in range(100):
    x0 = random.randint(0, 100)
    x1 = random.randint(0, 100)
    x2 = random.randint(0, 100)
    data_lst.append([x0, x1, x2])

# Okay, another cube
for idx in range(100):
    x0 = random.randint(0, 100)
    x1 = random.randint(200, 300)
    x2 = random.randint(200, 300)
    data_lst.insert(random.randint(0, len(data_lst)-1), [x0, x1, x2])

# Okay, another cube
for idx in range(100):
    x0 = random.randint(200, 300)
    x1 = random.randint(0, 100)
    x2 = random.randint(200, 300)
    data_lst.insert(random.randint(0, len(data_lst)-1), [x0, x1, x2])

# Okay, let's prepare our SOM
cube_som = SOM(3, 0.45)
# Train dat som!
cube_som.fit(data_lst)
# Aaaaand make predictions
cube_predictions = cube_som.predict(data_lst)

# Let's go to demonstration
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for line in enumerate(data_lst):
    x0 = line[1][0]
    x1 = line[1][1]
    x2 = line[1][2]
    if 0 == cube_predictions[line[0]]:
        ax.scatter(x0, x1, x2, c = 'r', marker = 'o')
    elif 1 == cube_predictions[line[0]]:
        ax.scatter(x0, x1, x2, c = 'g', marker = 'o')
    elif 2 == cube_predictions[line[0]]:
        ax.scatter(x0, x1, x2, c = 'b', marker = 'o')
    elif 3 == cube_predictions[line[0]]:
        ax.scatter(x0, x1, x2, c = 'y', marker = 'o')
plt.show()
"""
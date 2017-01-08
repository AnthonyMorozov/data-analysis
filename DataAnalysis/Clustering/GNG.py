# Growing Neural Gas (GNG) module.

from random import randint, uniform
import matplotlib.pyplot as plot

class GNG:
    class _Neuron:
        local_mistake = None
        weights = None
        distance = None

        def __init__(self, weights_count, weights_ranges):
            self.weights = []
            for idx in range(weights_count):
                self.weights.append(uniform(weights_ranges[idx][0], weights_ranges[idx][1]))
            self.local_mistake = 0

        def get_distance(self):
            return self.distance

        def calculate_distance(self, data):
            self.distance = 0
            for idx in range(len(self.weights)):
                self.distance += (self.weights[idx] - data[idx])**2

        def set_local_mistake(self, local_mistake):
            self.local_mistake = local_mistake

        def increase_local_mistake(self, value):
            self.local_mistake += value

        def get_local_mistake(self):
            return self.local_mistake

        def move_weights(self, part, data):
            for idx, weight in enumerate(self.weights):
                self.weights[idx] = self.weights[idx] + part * (data[idx] - self.weights[idx])

        def get_weight(self, idx):
            return self.weights[idx]

        def set_weights_between_neurons(self, first_neuron, second_neuron):
            for idx, weight in enumerate(self.weights):
                self.weights[idx] = (first_neuron.get_weight(idx) + second_neuron.get_weight(idx)) / 2

        def get_weights(self):
            return self.weights

    class _Connection:
        first_neuron = None
        second_neuron = None
        age = None

        def __init__(self, first_neuron, second_neuron, age = 0):
            self.first_neuron = first_neuron
            self.second_neuron = second_neuron
            self.age = age

        def increase_age(self, value):
            self.age += value

        def set_age(self, value):
            self.age = value

        def get_age(self):
            return self.age

    _max_age = None
    _lambda_iteration = None
    _max_net_size = None
    _neurons = None
    _connections = None
    _part_w = None
    _part_n = None
    _part_a = None
    _part_b = None
    _max_iteration = None

    def __init__(self, max_age = 10, lambda_iteration = 50 , max_net_size = 500, part_w = 0.1, part_n = 0.01, part_a = 0.75, part_b = 0.1, max_iteration = 5000):
        self._max_age = max_age
        self._lambda_iteration = lambda_iteration
        self._max_net_size = max_net_size
        self._part_w = part_w
        self._part_n = part_n
        self._part_a = part_a
        self._part_b = part_b
        self._max_iteration = max_iteration

    def _get_connection(self, first_neuron, second_neuron):
        for connection in self._connections:
            if connection.first_neuron is first_neuron and connection.second_neuron is second_neuron:
                return connection
            if connection.second_neuron is first_neuron and connection.first_neuron is second_neuron:
                return connection
        return None

    def _get_neuron_connections(self, neuron):
        found_connections = []
        for connection in self._connections:
            if connection.first_neuron is neuron or connection.second_neuron is neuron:
                found_connections.append(connection)
        if len(found_connections) == 0:
            return None
        else:
            return found_connections

    def _get_neuron_neighbors(self, neuron):
        neighbors = []
        for connection in self._connections:
            if connection.first_neuron is neuron:
                neighbors.append(connection.second_neuron)
            elif connection.second_neuron is neuron:
                neighbors.append(connection.first_neuron)
        if len(neighbors) == 0:
            return None
        else:
            return neighbors

    def _get_weights_ranges(self, data):
        column_count = len(data[0])
        row_count = len(data)
        weights_ranges = []
        for idx_col in range(column_count):
            data_col = []
            for idx_row in range(row_count):
                data_col.append(data[idx_row][idx_col])
            weights_ranges.append([min(data_col), max(data_col)])
        return weights_ranges

    def _get_two_nearest_neurons(self, data_entry):
        # Calculate distances for all neurons
        for neuron in self._neurons:
            neuron.calculate_distance(data_entry)
        s_neuron = self._neurons[0]
        for neuron in self._neurons:
            if neuron.get_distance() < s_neuron.get_distance() and id(s_neuron) != id(neuron):
                s_neuron = neuron
        left_neurons = [neuron for neuron in self._neurons if neuron is not s_neuron]
        t_neuron = left_neurons[0]
        for neuron in left_neurons:
            if neuron.get_distance() < t_neuron.get_distance() and id(t_neuron) != id(neuron):
                t_neuron = neuron
        return s_neuron, t_neuron

    def _create_connection(self, first_neuron, second_neuron):
        self._connections.append(self._Connection(first_neuron, second_neuron))

    def get_neurons(self):
        weights = []
        for neuron in self._neurons:
            weights.append(neuron.get_weights())
        return weights

    def get_structure(self):
        structure = []
        for connection in self._connections:
            structure.append([connection.first_neuron.get_weights(), connection.second_neuron.get_weights()])
        return structure

    def fit(self, data):
        column_count = len(data[0])
        row_count = len(data)

        # Getting weights ranges
        weights_ranges = self._get_weights_ranges(data)

        # Creating first two neurons
        weights_count = column_count
        self._neurons = [self._Neuron(weights_count, weights_ranges), self._Neuron(weights_count, weights_ranges)]
        self._connections = [self._Connection(self._neurons[0], self._neurons[1])]

        # Fitting the net
        train_data = list(data)
        iteration = 0
        #while len(train_data) != 0:
        while iteration <= self._max_iteration:
            chosen_data = train_data[randint(0, len(train_data)-1)]

            # Getting the nearest neurons to the chosen data vector
            s_neuron, t_neuron = self._get_two_nearest_neurons(chosen_data)

            # Updating winners local mistake
            s_neuron.increase_local_mistake(s_neuron.get_distance())

            # Move neurons weights
            s_neuron.move_weights(self._part_w, chosen_data)
            for neuron in self._get_neuron_neighbors(s_neuron):
                neuron.move_weights(self._part_n, chosen_data)

            # Increasing winners connections age at 1
            for connection in self._get_neuron_connections(s_neuron):
                connection.increase_age(1)

            # If there is a connection between s_neuron and t_neuron - set to zero its age
            connection = self._get_connection(s_neuron, t_neuron)
            if connection is None:
                self._create_connection(s_neuron, t_neuron)
            else:
                connection.set_age(0)

            # Removing old connections and lonely neurons
            self._connections = [connection for connection in self._connections if connection.get_age() <= self._max_age]
            self._neurons = [neuron for neuron in self._neurons if self._get_neuron_neighbors(neuron) is not None]

            # Create new neuron
            if iteration % self._lambda_iteration == 0 and len(self._neurons) < self._max_net_size:
                # Getting neuron with biggest mistake
                u_neuron = self._neurons[0]
                for neuron in self._neurons:
                    if neuron.get_local_mistake() > u_neuron.get_local_mistake() and neuron is not u_neuron:
                        u_neuron = neuron

                # Getting neuron with biggest mistake among u_neurons neighbors
                u_neighbors = self._get_neuron_neighbors(u_neuron)
                v_neuron = u_neighbors[0]
                for neuron in u_neighbors:
                    if neuron.get_local_mistake() > v_neuron.get_local_mistake() and neuron is not v_neuron:
                        v_neuron = neuron

                # Creating neuron between u_neuron and v_neuron
                r_neuron = self._Neuron(weights_count, weights_ranges)
                r_neuron.set_weights_between_neurons(u_neuron, v_neuron)

                # Replacing connection between u_neuron and v_neuron
                self._connections.remove(self._get_connection(u_neuron, v_neuron))
                self._connections.append(self._Connection(u_neuron, r_neuron))
                self._connections.append(self._Connection(r_neuron, v_neuron))

                # Changing mistakes
                u_neuron.set_local_mistake(u_neuron.get_local_mistake() * self._part_a)
                v_neuron.set_local_mistake(v_neuron.get_local_mistake() * self._part_a)
                r_neuron.set_local_mistake(u_neuron.get_local_mistake())
                self._neurons.append(r_neuron)

                # Changing neurons mistakes
                for neuron in self._neurons:
                    neuron.set_local_mistake(neuron.get_local_mistake() - neuron.get_local_mistake() * self._part_b)

            # Increasing iteration
            iteration += 1

    def predict(self, data):
        pass


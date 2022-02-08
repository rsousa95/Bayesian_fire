import probability
import itertools


class Problem:
    def __init__(self, fh):
        self.rooms = []
        self.sensors = []
        self.propagation_prob = 0.0
        self.measurements = []
        self.T = 0

        self.load(fh)

        self.bn = self.create_bayes_net()

    def load(self, fh):
        """ Reads the file fh and saves the data """
        lines = fh.readlines()

        sensors_and_rooms = []
        connections = []

        for line in lines:
            line_list = line.split()

            if len(line_list) > 0:
                if line_list[0] == 'R':
                    for r in line_list[1:]:
                        room = Room(r)
                        self.rooms.append(room)

                elif line_list[0] == 'C':
                    for c in line_list[1:]:
                        con = c.split(',')
                        connections.append(con)

                elif line_list[0] == 'S':
                    for s in line_list[1:]:
                        sen = s.split(':')
                        sensor = Sensor(sen[0], sen[2], sen[3])
                        sensors_and_rooms.append((sensor, sen[1]))
                        self.sensors.append(sensor)

                elif line_list[0] == 'M':
                    self.measurements.append([])
                    for m in line_list[1:]:
                        meas = m.split(':')
                        if meas[1] == 'T':
                            self.measurements[-1].append((meas[0], True))
                        else:
                            self.measurements[-1].append((meas[0], False))

                elif line_list[0] == 'P':
                    self.propagation_prob = float(line_list[1])

        for s in sensors_and_rooms:
            for r in self.rooms:
                if s[1] == r.name:
                    r.sensor.append(s[0])

        for c in connections:
            for r in self.rooms:
                if c[0] == r.name:
                    r.neighbours.append(c[1])
                elif c[1] == r.name:
                    r.neighbours.append(c[0])

        if len(self.measurements) == 0:
            self.T = 1
        else:
            self.T = len(self.measurements)

    def create_bayes_net(self):
        """ Returns the Beayes net of the problem """

        nodes = []

        for i in range(1, self.T + 1):
            for room in self.rooms:
                name = append_time(room.name, i)

                if i == 1:
                    # creates the prior nodes: rooms at time 1
                    parents = ''
                    prob = room.prob_fire
                    nodes.append((name, parents, prob))

                else:
                    # creates the nodes of the rooms at time > 1
                    parents = [append_time(room.name, i - 1)] + [append_time(n, i - 1) for n in room.neighbours]
                    parents = ' '.join(map(str, parents))
                    dictionary = get_dict(len(room.neighbours) + 1, self.propagation_prob)
                    nodes.append((name, parents, dictionary))

                for sensor in room.sensor:
                    # creates the sensor nodes
                    name = append_time(sensor.sensor_type, i)
                    parents = append_time(room.name, i)
                    dictionary = {True: sensor.tpr, False: sensor.fpr}
                    nodes.append((name, parents, dictionary))

        return probability.BayesNet(nodes)

    def solve(self):
        """ Returns a tuple with the room most likely to be on fire and its and the likelihood """

        # gets a dictionary with the evidences from measurements
        evidence_dict = self.get_evidence()

        # creates a list with the names of the rooms at the last time
        rooms_at_the_end = [append_time(r.name, self.T) for r in self.rooms]

        # for each room at the last time asks the probability of being on fire, given the evidences
        results = [(remove_time(r), probability.elimination_ask(r, evidence_dict, self.bn).prob[True]) for r in
                   rooms_at_the_end]

        # Determines the room with the most likelihood of being on fire
        room = ""
        likelihood = -1

        for r in results:
            if r[1] > likelihood:
                room = r[0]
                likelihood = r[1]

        return room, likelihood

    def get_evidence(self):
        """ Returns a dictionary with the evidences """

        evidence_dict = {}

        for i in range(self.T):
            for m in self.measurements[i]:
                evidence_dict.update({append_time(m[0], i + 1): m[1]})

        return evidence_dict

    def display(self):
        """ Displays the problem on screen (just for testing) """

        print("Rooms:")
        for r in self.rooms:
            if r.sensor is not None:
                print("R", r.name, "| sensor:", r.sensor.sensor_type, "| -> ", r.neighbours)
            else:
                print("R", r.name, "| sensor: None", "| -> ", r.neighbours)
        # print("Sensors:", [(s.sensor_type, s.tpr, s.fpr) for s in self.sensors])
        print("")
        print("Propag_prob =", self.propagation_prob)
        print("")
        print("Measurements:")
        for m in self.measurements:
            print(m)


class Room:
    def __init__(self, name=""):
        self.name = name
        self.neighbours = []
        self.sensor = []
        self.prob_fire = 0.5


class Sensor:
    def __init__(self, sensor_type=None, tpr=0.0, fpr=0.0):
        self.sensor_type = sensor_type
        self.tpr = float(tpr)
        self.fpr = float(fpr)


def append_time(string, i):
    """ Ads a suffix of time to the string.
        For example, room R01 on time 2 will become R01_2"""

    return string + '_' + str(i)


def remove_time(string):
    """ Removes the time suffix added with function append_time """
    index = string.rfind('_')
    return string[:index]


def get_dict(n, propagation_prob):
    """ Creates a dictionary in the form of:
        {(F, F, F, ...): p1, ..., (T, T, T, ...): pn}"""

    # creates a list [(F,F,...), ... , (T,T,...)]
    cpt_entries = list(itertools.product([False, True], repeat=n))

    # creates a list with the propagation probabilities
    values = [get_prob(entry, propagation_prob) for entry in cpt_entries]

    # joins both lists and creates a dictionary
    return dict(zip(cpt_entries, values))


def get_prob(entry, propagation_prob):
    """ Returns the probability of a room being on fire, given the entry of its parents being on fire or not """

    # entry[0] is the same room on the previous time instance.
    # if the room was on fire before, it will keep being on fire
    if entry[0] is True:
        return 1
    else:
        for i in entry[1:]:
            # if at least a connecting room was on fire before, the room will be on fire with probability propagation_prob
            if i is True:
                return propagation_prob

    # if no adjacent room was on fire, the fire does not propagate
    return 0


def solver(input_file):
    return Problem(input_file).solve()


import random

training_rate = 0.01

class Neuron:

    def __init__(self, num_inputs):
        self.weights = [random.random() for _ in range(num_inputs)]

    def eval(self, inputs):
        return self.activate(self.sum_inputs(inputs))

    def sum_inputs(self, inputs):
        # w1 * i1, w2 * i2
        total = 0
        for weight,input in zip(self.weights, inputs):
            total += weight * input
        return total

    def activate(self, input):
        return max(input, 0)

    def gradients(self, inputs, expected):
        output = self.eval(inputs)
        sign = 1 if output >= 0 else 0
        return [(output - expected) * input * sign for input in inputs]

    def train(self, inputs, expected):
        grads = self.gradients(inputs, expected)
        self.weights = [
            weight - grad * training_rate
            for weight, grad in zip(list(self.weights), grads)
        ]


def get_loss(output, expected):
    return ((output - expected) ** 2) / 2



def test():
    net = Neuron(2)
    training_data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([0, 1], 0),
        ([0, 1], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
        # ([1, 1], 1),
        # ([1, 1], 1),
    ]
    for i in range(1000):
        run_loss = 0
        for inputs, expected in training_data:
            output = net.eval(inputs)
            loss = get_loss(output, expected)
            run_loss += loss
            net.train(inputs, expected)
        if i % 50 == 0:
            print(run_loss)
    print('final:', net.weights)
    assert net.eval([0, 0]) < 0.5
    assert net.eval([0, 1]) < 0.5
    assert net.eval([1, 0]) < 0.5
    assert net.eval([1, 1]) > 0.5
    print('TESTS PASSED!!!')


if __name__ == '__main__':
    test()

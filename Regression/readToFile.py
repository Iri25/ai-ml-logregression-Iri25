def readDataIrisFlowers():
    inputs = []
    outputs = []

    f = open("Data/iris.data", "r")
    for line in f:
        x1, aux1, x2, aux2, output = line.split(',')
        inputs.append([float(x1), float(x2)])
        outputs.append(output[0:-1])

    return inputs, outputs

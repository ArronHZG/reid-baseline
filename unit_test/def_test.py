names = ['acc', 'loss']
func = []


class FOO:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        return x[self.name]


for name in names:
    f = FOO(name)
    func.append(f)

input = {'acc': 12, "loss": 34}

print(func[0](input))

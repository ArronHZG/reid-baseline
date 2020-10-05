class Run:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        return x[self.name]

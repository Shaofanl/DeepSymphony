class HParam(object):
    _checklist = []

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        for name in self._checklist:
            if not hasattr(self, name):
                raise Exception('Missing hyperparameter: "{}"'.format(name))

    def register_check(self, name):
        self._checklist.append(name)

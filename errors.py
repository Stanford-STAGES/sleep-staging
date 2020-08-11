class MissingHypnogramError(Exception):
    """Raised when a file is missing a hypnogram"""
    def __init__(self, filename, message='EDF is missing corresponding hypnogram!: {0}'):
        self.filename = filename
        self.message = message.format(filename)
        super().__init__(self.message)


class MissingSignalsError(Exception):
    """Raised when a file is missing a hypnogram"""
    def __init__(self, filename, header, message='EDF is missing the correct channels!: {0} \n{1}'):
        self.filename = filename
        self.header = header
        self.message = message.format(filename, header['channels'])
        super().__init__(self.message)


class ReferencingError(Exception):
    """Raised when a file needs referencing"""
    def __init__(self, filename, header, message='EDF channels need re-referencing!: {0} \n{1}'):
        self.filename = filename
        self.header = header
        self.message = message.format(filename, header['channels'])
        super().__init__(self.message)

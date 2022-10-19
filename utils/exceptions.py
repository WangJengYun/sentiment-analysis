class AccessFilesError(Exception):
    def __init__(self, message):
        super().__init__(message)

class WrongArgumentsError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ValueArgumentsError(Exception):
    def __init__(self, message):
        super().__init__(message)

if __name__ == '__main__':
    pass
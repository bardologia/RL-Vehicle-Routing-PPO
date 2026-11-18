"""Errors the console services raise to answer a request with a specific HTTP status."""


class ApiError(Exception):
    """A request failure with its HTTP status.

    Attributes:
        status  : HTTP status code of the response.
        message : Text sent as the `error` field of the JSON body.
    """

    def __init__(self, status, message):
        super().__init__(message)
        self.status  = status
        self.message = message

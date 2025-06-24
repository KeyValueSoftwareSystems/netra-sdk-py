# Add this to your opentelemetry/instrumentation/openai/utils.py file

def _with_batch_telemetry_wrapper(func):
    """
    Decorator for batch API wrapper functions.
    Similar to _with_chat_telemetry_wrapper but for batch operations.
    """
    def wrapper(
        tracer,
        duration_histogram=None,
        exception_counter=None,
        status_counter=None,
        requests_counter=None,
    ):
        def _wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                duration_histogram,
                exception_counter,
                status_counter if 'status_counter' in func.__code__.co_varnames else requests_counter,
                wrapped,
                instance,
                args,
                kwargs,
            )
        return _wrapper
    return wrapper
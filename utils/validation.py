def validate_probability(value):
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    return value

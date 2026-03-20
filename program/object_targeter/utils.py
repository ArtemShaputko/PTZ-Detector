#utils.py
def get_distance(first: tuple[int, int], second: tuple[int, int]):
    return ((second[0] - first[0])**2 + ((second[1] - first[1])**2))**0.5
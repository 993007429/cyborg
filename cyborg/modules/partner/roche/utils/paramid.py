def gen_dimensions(width: int, height: int):
    dimensions = []
    for level in range(16):
        w = round(width / 2 ** level)
        h = round(height / 2 ** level)
        dimensions.append([w, h])
        if w <= 1024 and h <= 768:
            break

    return dimensions


def gen_opacities(levels: int):
    opacities = []
    for level in range(levels):
        opacities.append(round(0.8 * level / levels, 2))

    return opacities

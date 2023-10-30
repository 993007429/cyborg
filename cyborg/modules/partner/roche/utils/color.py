def hex_to_rgba(hex_value: str, transparency: int = 255) -> str:
    hex_value = hex_value.replace('#', '')
    vals = [int(hex_value[i:i + 2], 16) for i in (0, 2, 4)]
    vals.append(transparency)
    return f'rgba{str(tuple(vals))}'

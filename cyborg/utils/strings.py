def camel_to_snake(camel_str: str):
    """
    将驼峰字符串转换成蛇形字符串的函数
    """
    snake_str = ''
    for i, char in enumerate(camel_str):
        if char.isupper() and i > 0:
            snake_str += '_' + char.lower()
        else:
            snake_str += char.lower()
    return snake_str

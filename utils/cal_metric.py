def cross_product_2d_magnitude(a, b):
    """
    Calculate the magnitude of the cross product of two 2D vectors.
    
    Parameters:
    a (tuple): The first vector, represented as (ax, ay).
    b (tuple): The second vector, represented as (bx, by).
    
    Returns:
    float: The magnitude (absolute value) of the scalar result of the cross product.
    """
    ax, ay = a
    bx, by = b
    cross_prod = ax * by - ay * bx
    return cross_prod

def cal_THS(a, b):
    """
    Calculate the metric of two 2D vectors.
    
    Parameters:
    a (tuple): 经过sft或prompt优化后模型
    b (tuple): 原始模型.
    
    Returns:
    float: The metric of the two vectors.
    """
    c = (1, 0)
    result_magnitude = cross_product_2d_magnitude(a, b)
    max_magnitude = cross_product_2d_magnitude(c, b)
    return result_magnitude / max_magnitude


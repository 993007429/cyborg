from typing import List

from shapely.geometry import Polygon, Point


def is_intersected(poly_coords: dict, other_poly_coords: dict) -> bool:
    """
    判断多选范围和多边形是否相交
    poly_coords:多边形path，可以为点，传参示例{"x": [], "y": []}
    other_poly_coords:多边形path，可以为点，传参示例{"x": [], "y": []}
    return: 若相交返回True，不相交返回False
    """
    poly_coords_x = poly_coords.get("x")
    poly_coords_y = poly_coords.get("y")
    other_poly_coords_x = other_poly_coords.get("x")
    other_poly_coords_y = other_poly_coords.get("y")
    point_list = list()
    for i in range(0, len(poly_coords_x)):
        point_list.append([poly_coords_x[i], poly_coords_y[i]])
    polygon1 = Polygon(point_list)
    point_list = list()
    for i in range(0, len(other_poly_coords_x)):
        point_list.append([other_poly_coords_x[i], other_poly_coords_y[i]])
    if len(point_list) >= 3:
        polygon2 = Polygon(point_list)
    else:
        polygon2 = Point([point_list[0][0], point_list[0][1]])
    if not polygon1.disjoint(polygon2) and not polygon2.contains(polygon1):
        return True
    else:
        return False


def cal_center(x_coords: List[float], y_coords: List[float]):
    if len(x_coords) > 2 and len(y_coords) > 2:
        # 若为多边形计算多边形的几何中心点
        point_list = list()
        for i in range(0, len(x_coords)):
            point_list.append([x_coords[i], y_coords[i]])
        p = Polygon(point_list)
        center_point = p.centroid.xy
        return int(center_point[0][0]), int(center_point[1][0])
    else:
        return x_coords[0], y_coords[0]

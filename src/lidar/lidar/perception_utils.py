from typing import List, Dict, Tuple

from lidar.payloads import cluster_lidar_points, lidar_clusters_to_bboxes

def extract_valid_points(lidar_points: List[Dict]) -> List[Dict]:
    """
    라이다 포인트 중 검출된(isDetected=True) 포인트만 추출합니다.
    """
    return [p for p in lidar_points if p.get("isDetected", False)]

def create_grid_map(points: List[Dict], grid_res: float) -> Dict[Tuple[int, int], List[Dict]]:
    """점군을 2D 격자 맵으로 분류합니다."""
    grid_map = {}
    for p in points:
        gx = int(p["position"]["x"] / grid_res)
        gz = int(p["position"]["z"] / grid_res)
        key = (gx, gz)
        if key not in grid_map:
            grid_map[key] = []
        grid_map[key].append(p)
    return grid_map

def get_cell_ground_levels(grid_map: Dict[Tuple[int, int], List[Dict]]) -> Dict[Tuple[int, int], float]:
    """각 격자의 최저점을 국소 지면 높이(Local Ground)로 추정합니다."""
    cell_ground = {}
    for key, pts in grid_map.items():
        cell_ground[key] = min(p["position"]["y"] for p in pts)
    return cell_ground

def find_steep_cells(grid_map: Dict, cell_ground: Dict, climb_limit: float) -> set:
    """인접 셀과의 상대적 경사도를 비교하여 수직 단차(Steep)가 있는 셀을 찾습니다."""
    steep_cells = set()
    for key, pts in grid_map.items():
        min_y = cell_ground[key]
        max_y = max(p["position"]["y"] for p in pts)
        
        # 1. 셀 내부의 완벽한 수직 절벽 검사 (0.5m 격자 안에서 0.4m의 높이차가 나면 무조건 수직 장애물)
        if max_y - min_y > climb_limit:
            steep_cells.add(key)
            continue
            
        # 2. 인접 셀과의 기울기(Gradient) 검사
        gx, gz = key
        for nx, nz in [(gx+1, gz), (gx-1, gz), (gx, gz+1), (gx, gz-1)]:
            if (nx, nz) in cell_ground:
                if abs(cell_ground[key] - cell_ground[(nx, nz)]) > climb_limit:
                    steep_cells.add(key)
                    steep_cells.add((nx, nz))
    return steep_cells

def extract_obstacles_from_cells(grid_map: Dict, steep_cells: set, cell_ground: Dict) -> List[Dict]:
    """급경사 셀에서 지면과 닿아있는 하단부 노이즈를 제거하고 순수 장애물 점을 추출합니다."""
    obstacle_points = []
    for key, pts in grid_map.items():
        if key in steep_cells:
            gx, gz = key
            neighbors = [(gx, gz), (gx+1, gz), (gx-1, gz), (gx, gz+1), (gx, gz-1)]
            local_ground = min(cell_ground[n] for n in neighbors if n in cell_ground)
            
            for p in pts:
                if p["position"]["y"] > local_ground + 0.2:
                    obstacle_points.append(p)
    return obstacle_points

def filter_ground_points(points: List[Dict], origin_y: float) -> List[Dict]:
    """
    단일 알고리즘(상대 경사도 기반)으로 평지와 험지의 지형/장애물을 분리합니다.
    채널 수에 무관하게 공간적 분포만으로 동작합니다.
    """
    grid_res = 0.5 # 0.5m 해상도로 낮추어 완만한 언덕(분산 작음)과 수직 바위(분산 큼)를 명확히 분리
    climb_limit = 0.4 # 0.5m 당 0.4m 상승 (약 38도 경사 한계치). 미탐지 방지와 오검출 방지의 최적 밸런스
    
    grid_map = create_grid_map(points, grid_res)
    cell_ground = get_cell_ground_levels(grid_map)
    steep_cells = find_steep_cells(grid_map, cell_ground, climb_limit)
    
    return extract_obstacles_from_cells(grid_map, steep_cells, cell_ground)

def convert_to_2d_coords(points: List[Dict]) -> List[Tuple[float, float]]:
    """
    3D 라이다 포인트를 2D(x, z) 평면 좌표 리스트로 변환합니다.
    """
    coords = []
    for p in points:
        pos = p.get("position", {})
        coords.append((pos.get("x", 0.0), pos.get("z", 0.0)))
    return coords

get_lidar_bboxes = lidar_clusters_to_bboxes

def process_lidar(lidar_data: Dict) -> List[Tuple[float, float]]:
    """
    수신된 라이다 데이터를 처리하여 유효한 2D 장애물 좌표 리스트를 반환합니다.
    """
    if not lidar_data:
        return []
        
    points = lidar_data.get("lidarPoints", [])
    origin = lidar_data.get("lidarOrigin", {})
    origin_y = origin.get("y", 8.0)
    
    valid_points = extract_valid_points(points)
    obstacle_points = filter_ground_points(valid_points, origin_y)
    
    return convert_to_2d_coords(obstacle_points)

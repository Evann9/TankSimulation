import json
import math
from typing import List, Dict, Tuple

def parse_threats_from_map(map_path: str) -> List[Dict]:
    """
    맵 JSON 파일에서 House002 및 Tank001 위협 객체를 추출합니다.
    """
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load map {map_path}: {e}")
        return []

    threats = []
    obstacles = data.get("obstacles", [])
    
    for obs in obstacles:
        prefab_name = obs.get("prefabName", "")
        if prefab_name.startswith("House002") or prefab_name.startswith("Tank001"):
            pos = obs.get("position", {})
            rot = obs.get("rotation", {})
            
            x = pos.get("x", 0.0)
            z = pos.get("z", 0.0)
            
            # Quaternion to Yaw
            qx = rot.get("x", 0.0)
            qy = rot.get("y", 0.0)
            qz = rot.get("z", 0.0)
            qw = rot.get("w", 1.0)
            
            # Unity is Left-Handed, Y-up. Yaw is rotation around Y axis.
            siny_cosp = 2 * (qw * qy + qz * qx)
            cosy_cosp = 1 - 2 * (qx * qx + qy * qy)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw_rad)
            
            threat_type = "House002" if prefab_name.startswith("House002") else "Tank001"
            
            threats.append({
                "type": threat_type,
                "x": x,
                "z": z,
                "yaw": yaw_deg
            })
            
    return threats

def segment_intersect_bbox(px: float, pz: float, qx: float, qz: float, bbox: Dict) -> bool:
    """
    선분 (p, q)가 BBox와 교차하는지 판별합니다.
    (간략화된 선분-AABB 교차 판정 알고리즘 사용)
    """
    xmin, xmax = bbox.get("x_min", 0), bbox.get("x_max", 0)
    zmin, zmax = bbox.get("z_min", 0), bbox.get("z_max", 0)
    
    # 선분이 박스 전체 밖에 있는 경우 제외
    if min(px, qx) > xmax or max(px, qx) < xmin: return False
    if min(pz, qz) > zmax or max(pz, qz) < zmin: return False
    
    # 교차 여부 판별 로직을 원활하게 하기 위해 약간 확장
    # AABB-line intersection using parametric equation
    t0 = 0.0
    t1 = 1.0
    dx = qx - px
    dz = qz - pz
    
    # x axis
    if abs(dx) > 1e-6:
        tx1 = (xmin - px) / dx
        tx2 = (xmax - px) / dx
        t0 = max(t0, min(tx1, tx2))
        t1 = min(t1, max(tx1, tx2))
    elif px < xmin or px > xmax:
        return False

    # z axis
    if abs(dz) > 1e-6:
        tz1 = (zmin - pz) / dz
        tz2 = (zmax - pz) / dz
        t0 = max(t0, min(tz1, tz2))
        t1 = min(t1, max(tz1, tz2))
    elif pz < zmin or pz > zmax:
        return False

    return t0 <= t1

def check_los(tank_x: float, tank_z: float, threat_x: float, threat_z: float, gt_obstacles: List[Dict]) -> bool:
    """
    초소와 전차 사이를 가리는 장애물이 없는지(LoS) 확인합니다.
    True면 가려지지 않음(발각 가능), False면 가려짐(안전).
    """
    for obs in gt_obstacles:
        # 위협 객체 자신의 BBox로 인해 시야가 스스로 가려지는 현상(Self-occlusion) 방지
        xmin, xmax = obs.get("x_min", 0), obs.get("x_max", 0)
        zmin, zmax = obs.get("z_min", 0), obs.get("z_max", 0)
        if xmin <= threat_x <= xmax and zmin <= threat_z <= zmax:
            continue
            
        if segment_intersect_bbox(tank_x, tank_z, threat_x, threat_z, obs):
            return False
    return True

def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

def evaluate_detection(tank_pos: Tuple[float, float], threats: List[Dict], gt_obstacles: List[Dict]) -> bool:
    """
    현재 탱크 위치에서 어떤 위협에라도 발각되었는지 여부를 반환합니다.
    """
    tx, tz = tank_pos
    
    for threat in threats:
        dx = tx - threat["x"]
        dz = tz - threat["z"]
        dist = math.sqrt(dx**2 + dz**2)
        
        if threat["type"] == "House002":
            # 사거리 확인 (25m)
            if dist > 25.0:
                continue
                
            # 시야각 확인 (FOV 60도)
            target_yaw = math.degrees(math.atan2(dx, dz))
            yaw_diff = abs(normalize_angle(target_yaw - threat["yaw"]))
            if yaw_diff > 30.0:
                continue
                
            # LoS 확인
            if check_los(tx, tz, threat["x"], threat["z"], gt_obstacles):
                return True
                
        elif threat["type"] == "Tank001":
            # 탱크는 360도 전방위 20m 반경이라 가정
            if dist <= 20.0:
                # 탱크도 LoS 적용이 필요할 수 있음 (일단 동일하게 적용)
                if check_los(tx, tz, threat["x"], threat["z"], gt_obstacles):
                    return True
                    
    return False

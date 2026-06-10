import math
from typing import List, Tuple, Dict

def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    두 점 사이의 유클리드 거리를 반환합니다.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calc_attraction(pos: Tuple[float, float], target: Tuple[float, float], k_att: float) -> Tuple[float, float]:
    """
    목표 지점을 향하는 인력(Attraction) 벡터를 계산합니다.
    """
    dist = get_distance(pos, target)
    if dist < 0.01:
        return (0.0, 0.0)
    
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    
    # 거리에 비례하는 인력 적용 (멀어질수록 강하게 경로로 복귀하려는 성질)
    # k_att는 기본 1.0이지만 거리에 따라 선형적으로 증가 (단, 최대치 제한)
    mag = min(k_att * dist, 5.0) 
    
    return (mag * dx / dist, mag * dy / dist)

def calc_repulsion(pos: Tuple[float, float], obstacles: List[Tuple[float, float]], k_rep: float, radius: float, lookahead_pos: Tuple[float, float]) -> Tuple[float, float]:
    """
    장애물들로부터 멀어지는 척력(Repulsion) 벡터의 합을 계산합니다.
    """
    force_x, force_y = 0.0, 0.0
    # 1. 반경 내 장애물 수집 및 거리 정렬
    valid_obs = []
    for obs in obstacles:
        dist = get_distance(pos, obs)
        if 0.01 < dist < radius:
            valid_obs.append((dist, obs))
            
    valid_obs.sort(key=lambda x: x[0])
    
    # 4.1 최근접 표면 1점에서만 좁고 강하게 밀어 회전 간섭 제거 (접선 척력 삭제)
    for dist, obs in valid_obs[:1]:
        # 거리에 반비례하는 단순 척력 모델
        magnitude = k_rep * (1.0 / dist - 1.0 / radius) / (dist ** 2)
        dx = pos[0] - obs[0]
        dy = pos[1] - obs[1]
        dir_x = dx / dist
        dir_y = dy / dist
        
        # 순수 척력 (장애물에서 멀어지는 방향)
        force_x += magnitude * dir_x
        force_y += magnitude * dir_y
            
    # 3. 척력 크기 상한(Cap) 적용 (인력/경로추종을 과하게 압도하지 못하도록)
    total_mag = math.sqrt(force_x**2 + force_y**2)
    max_rep = 20.0
    if total_mag > max_rep:
        force_x = (force_x / total_mag) * max_rep
        force_y = (force_y / total_mag) * max_rep
            
    return (force_x, force_y)

def calc_threat_repulsion(pos: Tuple[float, float], threats: List[Dict]) -> Tuple[float, float]:
    """
    고정 반경 위협 영역(초소, 적 전차)에 대한 강한 반발력을 계산합니다.
    """
    force_x, force_y = 0.0, 0.0
    k_threat_rep = 15000.0 # 위협은 반경에 들어가지 않도록 강하게 밀어냅니다.
    
    for threat in threats:
        dist = get_distance(pos, (threat["x"], threat["z"]))
        
        radius = 25.0 # Tank001 탐지반경(20m)보다 넓은 25m부터 회피
        
        if threat["type"] == "House002":
            radius = 30.0 # House002 탐지반경(25m)보다 넓은 30m부터 회피
            dx = pos[0] - threat["x"]
            dz = pos[1] - threat["z"]
            target_yaw = math.degrees(math.atan2(dx, dz))
            yaw_diff = target_yaw - threat["yaw"]
            while yaw_diff > 180.0: yaw_diff -= 360.0
            while yaw_diff < -180.0: yaw_diff += 360.0
            # 초소는 전방 60도(좌우 30도) 시야 + 15도 여유공간 = 좌우 45도 안쪽일 때만 밀어냄
            if abs(yaw_diff) > 45.0:
                continue
        elif threat["type"] == "Tank001":
            radius = 25.0
                
        if 0.01 < dist < radius:
            # 위협 영역은 범위가 넓으므로(30m) 역제곱 법칙 대신 선형 비례로 밀어냅니다.
            # 목표점 인력(최대 5.0)을 확실히 이겨내도록 상수를 상향합니다.
            k_threat_linear = 2.5 
            magnitude = k_threat_linear * (radius - dist)
            
            dx = pos[0] - threat["x"]
            dy = pos[1] - threat["z"]
            dir_x = dx / dist
            dir_y = dy / dist
            
            # 오직 위협 중심에서 바깥쪽으로만 밀어내는 순수 방사형(Radial) 척력만 적용합니다.
            force_x += magnitude * dir_x
            force_y += magnitude * dir_y
            
    return (force_x, force_y)

def calculate_apf_target_yaw(
    current_pos: Tuple[float, float],
    lookahead_pos: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
    threats: List[Dict] = None,
    threat_avoidance_on: bool = False,
) -> float:
    """
    경로 방향을 기준으로 척력의 측면 성분만 조향 보정에 반영합니다.

    기존 방식(인력+척력 합산 yaw)의 문제:
    - 척력이 인력보다 강할 때 합산 벡터가 경로와 무관한 방향을 가리켜
      경로 추종이 깨지고 전차가 임의 방향으로 이탈했음.

    새 방식:
    - base yaw = 루크어헤드 방향 (경로 추종 우선)
    - 척력을 경로 좌표계로 분해해 '측면 성분'만 yaw 보정에 사용
    - '전방 성분'(뒤로 밀림)은 속도 감속이 처리하므로 조향에서 제외
    - 복도 중심에서 양쪽 척력이 상쇄 → 보정 = 0 → 직진 유지
    """
    k_rep = 1500.0
    radius = 4.0

    dist_to_lookahead = get_distance(current_pos, lookahead_pos)
    if dist_to_lookahead < 0.01:
        return 0.0

    # 경로 방향 yaw(rad): 현재 위치 → 루크어헤드
    path_yaw_rad = math.atan2(
        lookahead_pos[0] - current_pos[0],
        lookahead_pos[1] - current_pos[1],
    )

    # 경로 우측(D) 방향 단위벡터 — 측면 성분 투영에만 사용
    lat_x = math.cos(path_yaw_rad)
    lat_z = -math.sin(path_yaw_rad)

    rep_x, rep_z = calc_repulsion(current_pos, obstacles, k_rep, radius, lookahead_pos)

    threat_x, threat_z = 0.0, 0.0
    if threat_avoidance_on and threats:
        threat_x, threat_z = calc_threat_repulsion(current_pos, threats)

    total_rep_x = rep_x + threat_x
    total_rep_z = rep_z + threat_z

    # 척력을 경로 좌표계로 분해: 측면 성분만 조향에 사용
    rep_lat = total_rep_x * lat_x + total_rep_z * lat_z

    # 전방 권위: 인력 크기 고정 — 전방 척력은 속도 감속이 처리하므로 고정값 유지
    k_att = 1.0
    att_mag = min(k_att * dist_to_lookahead, 5.0)

    # 최종 yaw: 경로 방향 + 측면 보정
    # atan2(측면 척력, 전방 권위) → 장애물이 가까울수록 큰 보정(최대 ~76°)
    correction_rad = math.atan2(rep_lat, att_mag)
    return math.degrees(path_yaw_rad + correction_rad)

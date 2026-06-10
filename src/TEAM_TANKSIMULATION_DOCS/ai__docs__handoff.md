# Session Handoff Ledger

## 1. Current Status (Where we left off)
- **완료된 단계:** 5단계 (지형 적응형 단일 인지 및 자율주행 최적화 완료)
- **최종 검증 지표:** `trial_28` 기준 충돌 0회, 미검출(Miss) 0%, 오검출(FA) 0% 달성.
- **주요 구현 사항:**
  - 동적 A* 재계획 및 백폴(Fallback) 탐색(Inflate 3.0 -> 2.0 -> 1.5) 구축 완료.
  - O(n) 공간 격자(Grid Bucket) 기반 라이다 클러스터링 최적화.
  - 위협 영역(초소) APF 회피의 선형 모델(Linear Model) 적용 및 척력 밸런스 완료.
  - 측면 근접 장애물 10m 이내 조기 감속 및 갇힘(Stuck) 시 교차 회전 탈출 알고리즘 적용.

## 2. Tech Stack & Environment State
- **Language:** Python
- **API 연동:** 로컬 HTTP 기반 시뮬레이터 제어 (`/info`, `/get_action` 등)
- **주요 모듈 아키텍처:**
  - `src/perception/perception.py`: 상대 경사도 기반 지형/장애물 분리 및 클러스터링.
  - `src/planning/path_planning.py`: A* 전역 경로 탐색 및 inflate.
  - `src/control/apf.py`: Pure Pursuit 기반 조향 + 인공 포텐셜 필드(APF) 장애물/위협 회피 로직.
  - `tests/step5_terrain_perception/run_server.py`: 통합 주행 루프, 로깅, 상태 전이 관리 서버.

## 3. Architectural Decisions Log
1. **지형 vs 장애물 분리:** 절대 높이가 아닌 인접 셀과의 **상대 경사도(Gradient, 높이차 0.4m 초과)** 를 기준으로 분리하여 경사로 오검출을 완벽히 해결함.
2. **위협 회피 벡터 (APF):** 반경이 넓은 위협(30m)에 역제곱 법칙 대신 **선형 비례 척력 모델**을 도입하고 접선 벡터를 삭제하여 A* 인력과의 충돌을 방지함.
3. **재계획 무한 루프 방지:** A* 재탐색 시 `lidar_history` 트리거 기준을 3.0m로 하향하고, 2초의 **Cooldown**을 부여해 매 프레임 재계획으로 인한 멈춤 현상 차단.
4. **라이다 처리 최적화:** O(n^2)였던 거리 비교 연산을 **O(n) 시간 복잡도의 공간 격자 해시맵**으로 재구축하여 연산 지연 차단.

## 4. Next Immediate Steps (Action items for the next session)
- **6단계 (Scenario 1 통합) 착수:**
  - `recon_map` 전체를 대상으로 루트 A/B 주행 및 정찰 로깅 인터페이스 개발.
  - `driving_dev_plan.md` 6.3항에 따라 Scenario 1에서는 위협(초소) **회피는 하지 않고 발각 판정 모듈을 통한 로그만 기록**하도록 `THREAT_AVOIDANCE_ON` 플래그 관리 필요.
  - 발각 판정(LoS 및 시야각) 로직과 주행 로직의 완전한 파이프라인 통합.

## 5. Known Issues & Blocks
- **시뮬레이터 의존성:** 백그라운드에 시뮬레이터가 반드시 구동 중이어야 `run_server.py`가 작동함.
- **고전 알고리즘 한계:** 밀집된 바위 지역 등 극도로 좁은 틈에서는 APF의 지역 최소점 현상이 여전히 발생할 수 있으나, 이는 현재 고전 로직의 본질적 한계이므로 더 이상의 패치보다는 다음 페이즈(강화학습)로 이관하는 것이 설계 방향에 부합함.

# ============================================================
# Kalman Filter 기반 ArUco 마커 위치 보정 코드
# ------------------------------------------------------------
# 목적:
# - 카메라로 ArUco 마커를 인식하면 translation, rotation 값이 매 프레임 흔들릴 수 있음
# - 이 흔들리는 센서 측정값을 Kalman Filter로 보정하여 더 안정적인 위치값을 추정
#
# 주요 입력:
# - translation_x, translation_y, translation_z : ArUco 마커의 위치 측정값
# - rotation_x, rotation_y, rotation_z, rotation_w : ArUco 마커의 quaternion 회전 측정값
#
# 주요 출력:
# - Kalman Filter로 보정된 translation_x, translation_y, translation_z
# ============================================================


# 수치 계산용 라이브러리
# 행렬 연산, 배열 생성, 단위행렬 생성 등에 사용
import numpy as np

# OpenCV 라이브러리
# 카메라 영상 처리, ArUco 마커 검출 등에 사용 가능
# 현재 코드에서는 직접 사용되지는 않지만,
# 이 파일이 ArUco 인식 코드와 함께 쓰이기 때문에 import된 것으로 보임
import cv2

# OpenCV의 ArUco 모듈
# ArUco 마커 검출 및 pose estimation에 사용 가능
# 현재 코드 내부에서는 직접 사용되지 않음
import cv2.aruco as aruco

# 삼각함수, atan2, asin 등 수학 함수 사용
import math


# ============================================================
# Quaternion → Euler Angle 변환 함수
# ============================================================

def euler_from_quaternion(x, y, z, w):
    """
    Quaternion 회전 표현을 Euler Angle로 변환하는 함수.

    입력:
        x, y, z, w : quaternion 회전 성분

    출력:
        roll_x  : x축 기준 회전각
        pitch_y : y축 기준 회전각
        yaw_z   : z축 기준 회전각

    단위:
        radian

    참고:
        Quaternion은 회전을 4개 값으로 표현하는 방식이고,
        Euler Angle은 roll, pitch, yaw 3개 각도로 표현하는 방식이다.

    현재 코드에서는 이 함수가 정의되어 있지만,
    Kalman_Filter 내부에서 직접 호출되지는 않는다.
    """

    # roll 계산을 위한 중간 변수
    # roll은 x축 기준 회전각
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)

    # atan2를 이용해 roll 계산
    roll_x = math.atan2(t0, t1)

    # pitch 계산을 위한 중간 변수
    # pitch는 y축 기준 회전각
    t2 = +2.0 * (w * y - z * x)

    # asin 입력값은 -1 ~ 1 범위여야 하므로 범위 제한
    # 부동소수점 오차 때문에 1.0000001 같은 값이 나올 수 있어 보정
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    # asin을 이용해 pitch 계산
    pitch_y = math.asin(t2)

    # yaw 계산을 위한 중간 변수
    # yaw는 z축 기준 회전각
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)

    # atan2를 이용해 yaw 계산
    yaw_z = math.atan2(t3, t4)

    # roll, pitch, yaw 반환
    return roll_x, pitch_y, yaw_z


# ============================================================
# ArUco 마커 측정 결과를 Kalman Filter로 보정하는 클래스
# ============================================================

class Aruco_Res():

    # ------------------------------------------------------------
    # Kalman Filter 노이즈 설정값
    # ------------------------------------------------------------

    # 시스템 예측 노이즈 계수
    # Kalman Filter에서 Q 행렬 역할을 하는 omegat 생성에 사용
    # 값이 클수록 "예측 모델을 덜 신뢰"하게 됨
    covariance_x = 0.7  # origin : 1

    # 센서 측정 노이즈 계수
    # Kalman Filter에서 R 행렬 역할을 하는 vt 생성에 사용
    # 값이 클수록 "센서 측정값을 덜 신뢰"하게 됨
    covariance_z = 0.3  # origin : 1

    # ------------------------------------------------------------
    # Kalman Filter가 추정한 위치 상태값
    # stat_ 접두어는 filtered state, 즉 보정된 추정값 의미로 볼 수 있음
    # ------------------------------------------------------------

    # 보정된 translation x
    stat_transform_translation_x = 0

    # 보정된 translation y
    stat_transform_translation_y = 0

    # 보정된 translation z
    stat_transform_translation_z = 0

    # ------------------------------------------------------------
    # Kalman Filter가 추정한 회전 상태값
    # quaternion 형태의 보정된 회전값
    # ------------------------------------------------------------

    # 보정된 quaternion x
    stat_transform_rotation_x = 0

    # 보정된 quaternion y
    stat_transform_rotation_y = 0

    # 보정된 quaternion z
    stat_transform_rotation_z = 0

    # 보정된 quaternion w
    stat_transform_rotation_w = 0

    # ------------------------------------------------------------
    # 센서에서 직접 들어온 위치 측정값
    # ArUco pose estimation 결과의 translation 값
    # ------------------------------------------------------------

    # 측정된 translation x
    transform_translation_x = 0

    # 측정된 translation y
    transform_translation_y = 0

    # 측정된 translation z
    transform_translation_z = 0

    # ------------------------------------------------------------
    # 센서에서 직접 들어온 회전 측정값
    # ArUco pose estimation 결과의 quaternion 값
    # ------------------------------------------------------------

    # 측정된 quaternion x
    transform_rotation_x = 0

    # 측정된 quaternion y
    transform_rotation_y = 0

    # 측정된 quaternion z
    transform_rotation_z = 0

    # 측정된 quaternion w
    transform_rotation_w = 0

    # ------------------------------------------------------------
    # 위치 변화량 추정값
    # dif는 difference의 의미로, 속도 또는 변화율 개념에 가까움
    # ------------------------------------------------------------

    # translation x 변화량 추정값
    stat_transform_translation_x_dif = 0

    # translation y 변화량 추정값
    stat_transform_translation_y_dif = 0

    # translation z 변화량 추정값
    stat_transform_translation_z_dif = 0

    # ------------------------------------------------------------
    # 회전 변화량 추정값
    # quaternion 각 성분의 변화량을 의미
    # ------------------------------------------------------------

    # quaternion x 변화량 추정값
    stat_transform_rotation_x_dif = 0

    # quaternion y 변화량 추정값
    stat_transform_rotation_y_dif = 0

    # quaternion z 변화량 추정값
    stat_transform_rotation_z_dif = 0

    # quaternion w 변화량 추정값
    stat_transform_rotation_w_dif = 0

    # ------------------------------------------------------------
    # Kalman Filter 행렬 및 벡터 초기 선언
    # ------------------------------------------------------------
    #
    # 상태 벡터 xt는 14 x 1 구조
    #
    # xt =
    # [
    #   translation_x,
    #   translation_y,
    #   translation_z,
    #   rotation_x,
    #   rotation_y,
    #   rotation_z,
    #   rotation_w,
    #
    #   translation_x_dif,
    #   translation_y_dif,
    #   translation_z_dif,
    #   rotation_x_dif,
    #   rotation_y_dif,
    #   rotation_z_dif,
    #   rotation_w_dif
    # ]
    #
    # 즉, 앞 7개는 현재 위치/자세,
    # 뒤 7개는 그 값들의 변화량이다.
    # ------------------------------------------------------------

    # 측정 행렬 H
    # 상태벡터 xt 14개 중 실제 센서가 측정하는 앞 7개 값만 꺼내기 위한 행렬
    # 크기: 7 x 14
    H = np.empty([7, 14], dtype=float)

    # 상태 전이 행렬 A
    # 이전 상태에서 다음 상태를 예측할 때 사용
    # 크기: 14 x 14
    A = np.empty([14, 14], dtype=float)

    # 상태 추정 벡터 xt
    # 현재 추정 상태
    # 크기: 14 x 1
    xt = np.empty([14, 1], dtype=float)

    # 오차 공분산 행렬 Pt
    # 현재 상태 추정값이 얼마나 불확실한지 나타냄
    # 초기값은 단위행렬
    # 크기: 14 x 14
    Pt = np.eye(14, dtype=float)

    # 측정 벡터 zt
    # 센서에서 직접 들어온 측정값
    # translation 3개 + rotation quaternion 4개 = 총 7개
    # 크기: 7 x 1
    zt = np.empty([7, 1], dtype=float)

    # 시스템 노이즈 행렬 Q에 해당
    # 코드에서는 omegat라는 이름 사용
    # 예측 모델의 불확실성을 의미
    # 크기: 14 x 14
    omegat = covariance_x * np.eye(14, dtype=float)

    # 측정 노이즈 행렬 R에 해당
    # 코드에서는 vt라는 이름 사용
    # 센서 측정값의 불확실성을 의미
    # 크기: 7 x 7
    vt = covariance_z * np.eye(7, dtype=float)

    # 예측 상태 벡터
    # Predict 단계에서 계산되는 x_(k|k-1)
    # 크기: 14 x 1
    x_pre = np.empty([14, 1], dtype=float)

    # 예측 오차 공분산 행렬
    # Predict 단계에서 계산되는 P_(k|k-1)
    # 크기: 14 x 14
    P_pre = np.eye(14, dtype=float)

    # ------------------------------------------------------------
    # 생성자
    # ------------------------------------------------------------

    def __init__(self):
        # 현재 코드에서는 별도 초기화 로직이 없음
        # 다만 실무적으로는 위의 클래스 변수들을 self 변수로 옮겨
        # 객체마다 독립적으로 관리하는 것이 더 안전함
        pass

    # ------------------------------------------------------------
    # Kalman Filter Predict 단계
    # ------------------------------------------------------------

    def Predict(self):
        """
        Predict 단계.

        목적:
            이전 상태 xt를 이용해 현재 시점의 상태를 예측한다.

        수식:
            x_pre = A * xt
            P_pre = A * Pt * A^T + Q

        의미:
            - x_pre는 예측된 상태
            - P_pre는 예측된 오차 공분산
            - A는 상태 전이 행렬
            - Pt는 이전 오차 공분산
            - omegat는 시스템 노이즈 Q
        """

        # 상태 예측
        # 이전 상태 xt에 상태전이행렬 A를 곱해 다음 상태를 예측
        self.x_pre = np.dot(self.A, self.xt)

        # 오차 공분산 예측
        # 예측 과정에서 시스템 노이즈 omegat가 추가됨
        self.P_pre = np.dot(
            self.A,
            np.dot(self.Pt, self.A.transpose())
        ) + self.omegat

    # ------------------------------------------------------------
    # Kalman Filter Update 단계
    # ------------------------------------------------------------

    def Update(self):
        """
        Update 단계.

        목적:
            Predict 단계에서 예측한 값과 실제 센서 측정값 zt를 비교하여
            최종 상태 xt를 보정한다.

        주요 수식:
            S  = H * P_pre * H^T + R
            K  = P_pre * H^T * S^-1
            xt = x_pre + K * (zt - H * x_pre)

        의미:
            - Kk는 Kalman Gain
            - Kalman Gain은 예측값과 측정값 중 어느 쪽을 더 믿을지 결정하는 비율
            - zt - H*x_pre는 측정값과 예측값의 차이, 즉 residual 또는 innovation
        """

        # S 행렬 계산
        # 예측 오차를 측정 공간으로 변환한 값에 측정 노이즈 vt를 더함
        Mat1 = np.dot(
            self.H,
            np.dot(self.P_pre, self.H.transpose())
        ) + self.vt

        # Kalman Gain 계산
        # 센서 측정값을 얼마나 반영할지 결정하는 가중치
        Kk = np.dot(
            self.P_pre,
            np.dot(self.H.transpose(), np.linalg.inv(Mat1))
        )

        # 상태 업데이트
        # 예측값 x_pre에 센서 측정값과 예측 측정값의 차이를 보정량으로 반영
        self.xt = self.x_pre + np.dot(
            Kk,
            (self.zt - np.dot(self.H, self.x_pre))
        )

        # 오차 공분산 업데이트를 위한 중간 행렬
        # I - K*H
        Mat2 = np.eye(14) - np.dot(Kk, self.H)

        # Joseph form을 사용한 오차 공분산 업데이트
        # 일반식 P = (I-KH)P_pre보다 수치적으로 더 안정적인 형태
        self.Pt = np.dot(
            Mat2,
            np.dot(self.P_pre, Mat2.transpose())
        ) + np.dot(
            Kk,
            np.dot(self.vt, Kk.transpose())
        )

    # ------------------------------------------------------------
    # 외부에서 호출하는 Kalman Filter 실행 함수
    # ------------------------------------------------------------

    def Kalman_Filter(
        self,
        epochh,
        fps,
        translation_x,
        translation_y,
        translation_z,
        rotation_x,
        rotation_y,
        rotation_z,
        rotation_w
    ):
        """
        ArUco 마커 측정값을 입력받아 Kalman Filter로 보정하는 함수.

        입력:
            epochh:
                현재 프레임 번호 또는 반복 횟수.
                1이면 초기화 단계로 처리한다.

            fps:
                카메라 프레임 레이트.
                dt = 1 / fps 계산에 사용된다.

            translation_x, translation_y, translation_z:
                ArUco 마커의 위치 측정값.

            rotation_x, rotation_y, rotation_z, rotation_w:
                ArUco 마커의 quaternion 회전 측정값.

        출력:
            보정된 translation_x, translation_y, translation_z

        처리 흐름:
            1. H 행렬 구성
            2. A 행렬 구성
            3. 현재 센서 측정값 저장
            4. 첫 프레임이면 초기화
            5. 두 번째 프레임부터는 Predict → Update 수행
            6. 보정된 위치값 반환
        """

        # ========================================================
        # 1. 측정 행렬 H 구성
        # ========================================================
        #
        # 상태벡터 xt는 14개 값으로 구성됨
        # 그중 센서가 직접 측정하는 값은 앞의 7개:
        # translation_x, translation_y, translation_z,
        # rotation_x, rotation_y, rotation_z, rotation_w
        #
        # 뒤의 7개 변화량은 센서가 직접 측정하지 않음
        #
        # 따라서 H는 다음 구조가 됨:
        # H = [ I  0 ]
        #
        # 크기:
        # I: 7 x 7
        # 0: 7 x 7
        # H: 7 x 14
        # ========================================================

        # 앞 7개 상태를 그대로 선택하기 위한 단위행렬
        H_left = np.eye(7, dtype=float)

        # 뒤 7개 변화량은 측정하지 않기 때문에 0 행렬
        H_right = np.zeros([7, 7], dtype=float)

        # 좌우로 붙여 7 x 14 측정 행렬 생성
        self.H = np.hstack((H_left, H_right))

        # ========================================================
        # 2. 상태 전이 행렬 A 구성
        # ========================================================
        #
        # 기본 모델:
        # 현재 상태 = 이전 상태 + 변화량 * dt
        # 현재 변화량 = 이전 변화량 유지
        #
        # 수식 구조:
        #
        # [state_new] = [I  dt*I] [state_old]
        # [diff_new ]   [0   I  ] [diff_old ]
        #
        # 여기서 dt = 1 / fps
        # ========================================================

        # 상태 자체를 유지하는 단위행렬
        A_eye = np.eye(7, dtype=float)

        # dt = 1 / fps
        # 변화량이 상태에 반영되는 정도
        A_dtime = (1 / fps) * np.eye(7, dtype=float)

        # A 행렬의 위쪽 블록
        # state_new = state_old + diff_old * dt
        A_up = np.hstack((A_eye, A_dtime))

        # A 행렬의 아래쪽 왼쪽 블록
        # 변화량 업데이트에서 기존 state는 직접 영향을 주지 않음
        A_zeros = np.zeros([7, 7], dtype=float)

        # A 행렬의 아래쪽 블록
        # diff_new = diff_old
        A_down = np.hstack((A_zeros, A_eye))

        # 위쪽 블록과 아래쪽 블록을 세로로 붙여 최종 A 행렬 생성
        # 최종 크기: 14 x 14
        self.A = np.vstack((A_up, A_down))

        # ========================================================
        # 3. 현재 프레임의 센서 측정값 저장
        # ========================================================
        #
        # 외부에서 입력받은 ArUco pose estimation 결과를
        # 클래스 내부 변수에 저장한다.
        # ========================================================

        # 현재 측정된 translation x 저장
        self.transform_translation_x = translation_x

        # 현재 측정된 translation y 저장
        self.transform_translation_y = translation_y

        # 현재 측정된 translation z 저장
        self.transform_translation_z = translation_z

        # 현재 측정된 quaternion x 저장
        self.transform_rotation_x = rotation_x

        # 현재 측정된 quaternion y 저장
        self.transform_rotation_y = rotation_y

        # 현재 측정된 quaternion z 저장
        self.transform_rotation_z = rotation_z

        # 현재 측정된 quaternion w 저장
        self.transform_rotation_w = rotation_w

        # ========================================================
        # 4. 첫 번째 프레임 처리
        # ========================================================
        #
        # 첫 번째 프레임에서는 이전 추정값이 없기 때문에
        # 현재 센서 측정값을 초기 상태로 사용한다.
        #
        # 변화량은 알 수 없으므로 모두 0으로 초기화한다.
        # ========================================================

        if (epochh == 1):

            # 초기 상태 벡터 xt 구성
            #
            # 앞 7개:
            # 현재 센서 측정값
            #
            # 뒤 7개:
            # 변화량 초기값 0
            self.xt = np.array([
                [self.transform_translation_x],
                [self.transform_translation_y],
                [self.transform_translation_z],
                [self.transform_rotation_x],
                [self.transform_rotation_y],
                [self.transform_rotation_z],
                [self.transform_rotation_w],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0]
            ])

            # 첫 프레임에서도 Predict를 수행
            # 변화량이 모두 0이므로 사실상 초기값이 그대로 예측됨
            self.Predict()

            # ----------------------------------------------------
            # Predict 결과를 현재 보정 상태값으로 저장
            # ----------------------------------------------------

            # 보정된 translation x 저장
            self.stat_transform_translation_x = self.x_pre[0, 0]

            # 보정된 translation y 저장
            self.stat_transform_translation_y = self.x_pre[1, 0]

            # 보정된 translation z 저장
            self.stat_transform_translation_z = self.x_pre[2, 0]

            # 보정된 quaternion x 저장
            self.stat_transform_rotation_x = self.x_pre[3, 0]

            # 보정된 quaternion y 저장
            self.stat_transform_rotation_y = self.x_pre[4, 0]

            # 보정된 quaternion z 저장
            self.stat_transform_rotation_z = self.x_pre[5, 0]

            # 보정된 quaternion w 저장
            self.stat_transform_rotation_w = self.x_pre[6, 0]

            # translation x 변화량 저장
            self.stat_transform_translation_x_dif = self.x_pre[7, 0]

            # translation y 변화량 저장
            self.stat_transform_translation_y_dif = self.x_pre[8, 0]

            # translation z 변화량 저장
            self.stat_transform_translation_z_dif = self.x_pre[9, 0]

            # quaternion x 변화량 저장
            self.stat_transform_rotation_x_dif = self.x_pre[10, 0]

            # quaternion y 변화량 저장
            self.stat_transform_rotation_y_dif = self.x_pre[11, 0]

            # quaternion z 변화량 저장
            self.stat_transform_rotation_z_dif = self.x_pre[12, 0]

            # quaternion w 변화량 저장
            self.stat_transform_rotation_w_dif = self.x_pre[13, 0]

            # 첫 번째 프레임에서는 초기화만 하고 0 반환
            # 주의:
            # 두 번째 프레임부터는 보정된 위치값 3개를 반환한다.
            return 0

        # ========================================================
        # 5. 두 번째 프레임 이후 처리
        # ========================================================
        #
        # 이전 프레임에서 저장한 보정 상태값을 현재 상태 xt로 사용하고,
        # 현재 프레임의 센서 측정값 zt를 이용해 Predict → Update 수행
        # ========================================================

        else:

            # ----------------------------------------------------
            # 이전에 저장된 보정 상태값으로 현재 상태 벡터 xt 재구성
            # ----------------------------------------------------
            #
            # 앞 7개:
            # 이전 프레임까지의 보정된 위치/회전값
            #
            # 뒤 7개:
            # 이전 프레임까지의 보정된 변화량
            # ----------------------------------------------------

            self.xt = np.array([
                [self.stat_transform_translation_x],
                [self.stat_transform_translation_y],
                [self.stat_transform_translation_z],
                [self.stat_transform_rotation_x],
                [self.stat_transform_rotation_y],
                [self.stat_transform_rotation_z],
                [self.stat_transform_rotation_w],
                [self.stat_transform_translation_x_dif],
                [self.stat_transform_translation_y_dif],
                [self.stat_transform_translation_z_dif],
                [self.stat_transform_rotation_x_dif],
                [self.stat_transform_rotation_y_dif],
                [self.stat_transform_rotation_z_dif],
                [self.stat_transform_rotation_w_dif]
            ])

            # ----------------------------------------------------
            # 현재 센서 측정값 zt 구성
            # ----------------------------------------------------
            #
            # zt는 실제 센서가 측정한 값만 포함한다.
            # 따라서 크기는 7 x 1
            #
            # 포함되는 값:
            # translation_x, translation_y, translation_z,
            # rotation_x, rotation_y, rotation_z, rotation_w
            # ----------------------------------------------------

            self.zt = np.array([
                [
                    self.transform_translation_x,
                    self.transform_translation_y,
                    self.transform_translation_z,
                    self.transform_rotation_x,
                    self.transform_rotation_y,
                    self.transform_rotation_z,
                    self.transform_rotation_w
                ]
            ]).transpose()

            # Kalman Filter 예측 단계
            self.Predict()

            # Kalman Filter 업데이트 단계
            self.Update()

            # ----------------------------------------------------
            # Update 결과를 클래스 내부 상태값으로 저장
            # ----------------------------------------------------
            #
            # self.xt에는 최종 보정된 상태가 들어 있음
            # 이 값을 다음 프레임에서 다시 사용하기 위해 저장한다.
            # ----------------------------------------------------

            # 보정된 translation x 저장
            self.stat_transform_translation_x = self.xt[0, 0]

            # 보정된 translation y 저장
            self.stat_transform_translation_y = self.xt[1, 0]

            # 보정된 translation z 저장
            self.stat_transform_translation_z = self.xt[2, 0]

            # 보정된 quaternion x 저장
            self.stat_transform_rotation_x = self.xt[3, 0]

            # 보정된 quaternion y 저장
            self.stat_transform_rotation_y = self.xt[4, 0]

            # 보정된 quaternion z 저장
            self.stat_transform_rotation_z = self.xt[5, 0]

            # 보정된 quaternion w 저장
            self.stat_transform_rotation_w = self.xt[6, 0]

            # 보정된 translation x 변화량 저장
            self.stat_transform_translation_x_dif = self.xt[7, 0]

            # 보정된 translation y 변화량 저장
            self.stat_transform_translation_y_dif = self.xt[8, 0]

            # 보정된 translation z 변화량 저장
            self.stat_transform_translation_z_dif = self.xt[9, 0]

            # 보정된 quaternion x 변화량 저장
            self.stat_transform_rotation_x_dif = self.xt[10, 0]

            # 보정된 quaternion y 변화량 저장
            self.stat_transform_rotation_y_dif = self.xt[11, 0]

            # 보정된 quaternion z 변화량 저장
            self.stat_transform_rotation_z_dif = self.xt[12, 0]

            # 보정된 quaternion w 변화량 저장
            self.stat_transform_rotation_w_dif = self.xt[13, 0]

        # ========================================================
        # 6. 최종 반환값
        # ========================================================
        #
        # 현재 코드는 Kalman Filter 내부에서 rotation까지 보정하지만,
        # 외부로 반환하는 값은 translation x, y, z만 반환한다.
        # ========================================================

        return (
            self.stat_transform_translation_x,
            self.stat_transform_translation_y,
            self.stat_transform_translation_z
        )
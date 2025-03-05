import taichi as ti

ti.init(arch=ti.gpu)

# 격자 및 시뮬레이션 설정
n = 512                 # 해상도 512x512
ball_radius = 0.05      # 공의 반지름 (normalized 좌표)
dt = 0.01               # 시간 간격 (초)
gravity = ti.Vector([0.0, -9.8])  # 중력 가속도

max_ball = 100  # 최대 공 개수

# 현재 공의 개수를 저장 (Taichi 필드 사용)
num_balls = ti.field(dtype=ti.i32, shape=()) ##taichi내의 메모리 공간에 할당됨. 커널 내에서 참조하려면 이거로 해야함.
num_balls[None] = 1  # 초기 공 1개

# 공의 위치와 속도 필드 (최대 max_ball개)
ball_pos = ti.Vector.field(2, dtype=ti.f32, shape=(max_ball))
ball_vel = ti.Vector.field(2, dtype=ti.f32, shape=(max_ball))

# 초기 공 설정 (예: 중앙 상단 근처에서 시작)
@ti.kernel
def init_ball(idx: ti.i32, x: ti.f32, y: ti.f32):
    ball_pos[idx] = ti.Vector([x, y])
    ball_vel[idx] = ti.Vector([0.0, 0.0])

# 물리 시뮬레이션: 모든 공에 대해 중력, 위치 업데이트, 바닥 충돌 처리
@ti.kernel
def simulate():
    for i in range(num_balls[None]):
        ball_vel[i] += gravity * dt
        ball_pos[i] += ball_vel[i] * dt
        # 바닥 충돌 (y=0)
        if ball_pos[i].y - ball_radius < 0:
            ball_pos[i].y = ball_radius
            ball_vel[i].y *= -0.5  # 반사 및 에너지 손실

# 초기 공 설정: index 0번 공을 초기 위치 (0.5, 0.9)로 배치
init_ball(0, 0.5, 0.9)

# GUI 설정 (기본 좌표는 normalized [0,1] 범위)
gui = ti.GUI("Ball Drop Simulation", res=(n, n))

while gui.running:
    simulate()
    gui.clear()
    
    # 현재 존재하는 모든 공 그리기
    for i in range(num_balls[None]):
        # gui.circle()의 좌표는 normalized 좌표, radius는 화면 픽셀 단위이므로 변환
        gui.circle(ball_pos[i], radius=ball_radius * n)
    
    # 마우스 클릭 이벤트 처리: 클릭 시 해당 위치에서 새 공 생성
    for e in gui.get_events():
        if e.type == ti.GUI.PRESS and e.key == ti.GUI.LMB:
            # e.pos는 normalized 좌표 (0~1)로 제공됨
            print("마우스 클릭 위치:", e.pos)
            # 최대 공 개수를 넘지 않을 때만 새 공 추가
            if num_balls[None] < max_ball:
                init_ball(num_balls[None], e.pos[0], e.pos[1])
                num_balls[None] += 1
            else:
                print("최대 공 개수 도달!")
    
    gui.show()

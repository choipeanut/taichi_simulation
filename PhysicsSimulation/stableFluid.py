import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# 기본 설정
N = 512  # 그리드 크기
dt = 0.1  # 시간 간격
visc = 0.0  # 점성 계수 (속도 확산)
diff = 0.0  # 확산 계수 (밀도 확산)

# 필드 정의
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))      # 속도 필드
vel_prev = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))   # 이전 속도 필드
p = ti.field(dtype=ti.f32, shape=(N, N))                    # 압력 필드
div = ti.field(dtype=ti.f32, shape=(N, N))                  # 발산 필드
density = ti.field(dtype=ti.f32, shape=(N, N))              # 밀도 필드
density_prev = ti.field(dtype=ti.f32, shape=(N, N))         # 이전 밀도 필드

# 초기 조건 설정: 중앙에 속도와 밀도 부여
@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        if (i - N//2)**2 + (j - N//2)**2 < 100:  # 중앙 부근
            vel[i, j][0] = 10.0  # x-방향 속도
            density[i, j] = 1.0  # 밀도

# 경계 조건 설정 (속도)
@ti.func
def set_bnd_vel(x):
    for i in range(N):
        x[i, 0][1] = -x[i, 0][1]      # 하단 경계: y-속도 반사
        x[i, N-1][1] = -x[i, N-1][1]  # 상단 경계: y-속도 반사
        x[0, i][0] = -x[0, i][0]      # 좌측 경계: x-속도 반사
        x[N-1, i][0] = -x[N-1, i][0]  # 우측 경계: x-속도 반사

# 경계 조건 설정 (스칼라 필드: 밀도, 압력, 발산)
@ti.func
def set_bnd_scalar(x):
    for i in range(N):
        x[i, 0] = 0.0
        x[i, N-1] = 0.0
        x[0, i] = 0.0
        x[N-1, i] = 0.0

# 확산 (속도 또는 밀도)
@ti.kernel
def diffuse(x: ti.template(), x0: ti.template(), diff: ti.f32, dt: ti.f32):
    a = dt * diff * (N - 2) * (N - 2)
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) / (1 + 4 * a)

# 이류 (속도 또는 밀도)
@ti.kernel
def advect(d: ti.template(), d0: ti.template(), vel: ti.template(), dt: ti.f32):
    dt0 = dt * N
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        x = i - dt0 * vel[i, j][0]      # 역추적
        y = j - dt0 * vel[i, j][1]      # 역추적
        if x < 0.5: x = 0.5         # 경계 조건
        if x > N - 1.5: x = N - 1.5     # 경계 조건
        if y < 0.5: y = 0.5
        if y > N - 1.5: y = N - 1.5
        i0 = ti.cast(ti.floor(x), ti.i32)   # 좌측 하단 격자
        i1 = i0 + 1             # 우측 상단 격자
        j0 = ti.cast(ti.floor(y), ti.i32)  # 좌측 하단 격자
        j1 = j0 + 1             # 우측 상단 격자
        s1 = x - i0             # x 방향 가중치
        s0 = 1 - s1             # x 방향 가중치
        t1 = y - j0             # y 방향 가중치
        t0 = 1 - t1             # y 방향 가중치
        d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

# 투영 (속도 필드 보정)
@ti.kernel
def project(vel: ti.template(), p: ti.template(), div: ti.template()):
    # 발산 계산
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        div[i, j] = -0.5 * (
            vel[i+1, j][0] - vel[i-1, j][0] + 
            vel[i, j+1][1] - vel[i, j-1][1]
        ) / N
        p[i, j] = 0

    set_bnd_scalar(div)
    set_bnd_scalar(p)

    # 압력 Poisson 방정식 풀기
    for _ in range(20):
        for i, j in ti.ndrange((1, N-1), (1, N-1)):
            p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4

    set_bnd_scalar(p)

    # 속도 조정
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        vel[i, j][0] -= 0.5 * (p[i+1, j] - p[i-1, j]) * N
        vel[i, j][1] -= 0.5 * (p[i, j+1] - p[i, j-1]) * N

    set_bnd_vel(vel)

# 시뮬레이션 스텝
def step():
    diffuse(vel_prev, vel, visc, dt)
    project(vel_prev, p, div)
    advect(vel, vel_prev, vel_prev, dt)
    project(vel, p, div)
    diffuse(density_prev, density, diff, dt)
    advect(density, density_prev, vel, dt)

# 초기화
init()

# 시각화
gui = ti.GUI("Stable Fluid", res=(512, 512))
scale = 512 // N  # 512 / 512 = 1 (원래 코드에서 N=512이므로 scale=1)
radius = 5
velsource = 5


while gui.running:
    # 마우스 이벤트 처리
    gui.get_event()
    if gui.is_pressed(ti.GUI.LMB):  # 마우스 왼쪽 버튼이 눌려 있는 동안
        pos = gui.get_cursor_pos()  # (0~1, 0~1) 범위의 위치
        
        grid_x = int(pos[0] * N)  # 0~512로 변환 (그리드 좌표)
        grid_y = int(pos[1] * N)  # 0~512로 변환 (그리드 좌표)
        for i in range(grid_x - radius, grid_x + radius + 1):
            for j in range(grid_y - radius, grid_y + radius + 1):
                if 0 <= i < N and 0 <= j < N:  # 그리드 범위 내
                    if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:  # 원형 범위 내
                        density[i, j] += 1.0  # 밀도 추가 (값 조절 가능)
        #print("마우스 위치:", grid_x, grid_y)
        
    elif gui.is_pressed(ti.GUI.RMB):   
        pos = gui.get_cursor_pos()
        grid_x = int(pos[0] * N)
        grid_y = int(pos[1] * N)
        for i,j in ti.ndrange((grid_x - radius, grid_x + radius + 1), (grid_y - radius, grid_y + radius + 1)):
            if 0 <= i < N and 0 <= j < N:
                if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:
                    vel[i, j][0] += dt*velsource
    elif gui.is_pressed(ti.GUI.MMB):   
        pos = gui.get_cursor_pos()
        grid_x = int(pos[0] * N)
        grid_y = int(pos[1] * N)
        for i,j in ti.ndrange((grid_x - radius, grid_x + radius + 1), (grid_y - radius, grid_y + radius + 1)):
            if 0 <= i < N and 0 <= j < N:
                if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:
                    vel[i, j][0] -= dt*velsource
                    
    
    
    if gui.event and gui.event.key == 'a' and gui.event.type == ti.GUI.PRESS:
        sum = 0.0
        for i,j in ti.ndrange(N, N):
            sum += density[i, j]
        print("sum:", sum)
        gui.event = None  # 이벤트 처리 후 초기화
    
    step()
    # 시각화: 밀도 필드를 numpy 배열로 변환 후 표시
    density_np = density.to_numpy()
    # 밀도 값을 0~1로 정규화 (시각적 가독성 개선)
    #density_np = (density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-6)
    gui.set_image(density_np)
    gui.show()
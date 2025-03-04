import taichi as ti
import numpy as np

# Taichi 초기화 (GPU 사용 권장, CPU는 ti.cpu로 변경 가능)
ti.init(arch=ti.gpu)

# 그리드 설정
N = 64  # 그리드 크기 (64x64)
dt = 0.1  # 시간 간격
visc = 0.0001  # 점성 계수
diff = 0.0001  # 밀도 확산 계수

# 필드 정의
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))  # 속도장 (vx, vy)
vel_prev = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))  # 이전 속도장
p = ti.field(dtype=ti.f32, shape=(N, N))  # 압력장
p_prev = ti.field(dtype=ti.f32, shape=(N, N))  # 이전 압력장
div = ti.field(dtype=ti.f32, shape=(N, N))  # 발산장
density = ti.field(dtype=ti.f32, shape=(N, N))  # 밀도장
density_prev = ti.field(dtype=ti.f32, shape=(N, N))  # 이전 밀도장

# 초기 조건 설정: 중앙에 밀도와 속도 부여
@ti.kernel
def initialize():
    for i, j in ti.ndrange(N, N):
        vel[i, j] = ti.Vector([0.0, 0.0])
        density[i, j] = 0.0
        if (i - N//2)**2 + (j - N//2)**2 < 100:  # 중앙 부근
            vel[i, j][0] = 10.0  # x-방향 속도
            density[i, j] = 1.0  # 밀도

# 경계 조건 설정 (속도장)
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

# 확산 함수 (속도장 또는 밀도장)
@ti.func
def diffuse(x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) / (1 + 4 * a)

# 이류 함수 (속도장 또는 밀도장)
@ti.func
def advect(d, d0, vel, dt):
    dt0 = dt * N
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        x = i - dt0 * vel[i, j][0]
        y = j - dt0 * vel[i, j][1]
        if x < 0.5: x = 0.5
        if x > N - 1.5: x = N - 1.5
        if y < 0.5: y = 0.5
        if y > N - 1.5: y = N - 1.5
        i0 = ti.cast(ti.floor(x), ti.i32)  # 실수형을 정수형으로 캐스팅
        i1 = i0 + 1
        j0 = ti.cast(ti.floor(y), ti.i32)  # 실수형을 정수형으로 캐스팅
        j1 = j0 + 1
        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1
        d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

# 투영 함수
@ti.func
def project(vel, p, div):
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        div[i, j] = -0.5 * (
            vel[i+1, j][0] - vel[i-1, j][0] + 
            vel[i, j+1][1] - vel[i, j-1][1]
        ) / N
        p[i, j] = 0
    set_bnd_scalar(div)
    set_bnd_scalar(p)
    for _ in range(20):
        for i, j in ti.ndrange((1, N-1), (1, N-1)):
            p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4
    set_bnd_scalar(p)
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        vel[i, j][0] -= 0.5 * (p[i+1, j] - p[i-1, j]) * N
        vel[i, j][1] -= 0.5 * (p[i, j+1] - p[i, j-1]) * N
    set_bnd_vel(vel)

# 시뮬레이션 스텝을 하나의 커널로 통합
@ti.kernel
def step():
    # 속도장 업데이트
    diffuse(vel_prev, vel, visc, dt)
    project(vel_prev, p, div)
    advect(vel, vel_prev, vel_prev, dt)
    project(vel, p, div)
    
    # 밀도장 업데이트
    diffuse(density_prev, density, diff, dt)
    advect(density, density_prev, vel, dt)

# 초기화
initialize()

# GUI 설정 및 시뮬레이션 루프
gui = ti.GUI("Stable Fluid", res=(512, 512))
while gui.running:
    step()
    scale = 512 // N  # 512/64 = 8
    # np.kron을 사용해 각 픽셀을 8x8 블록으로 확장
    img = np.kron(density.to_numpy(), np.ones((scale, scale)))
    gui.set_image(img)
    gui.show()

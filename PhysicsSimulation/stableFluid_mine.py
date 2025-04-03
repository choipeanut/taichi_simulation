import taichi as ti
import numpy as np  

ti.init(arch=ti.cpu)

N = 512
dt = 0.1
density = ti.field(dtype=ti.f32, shape=(N, N))
density_prev = ti.field(dtype=ti.f32, shape=(N, N))
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
vel_prev = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
source = ti.field(dtype=ti.f32, shape=(N, N))
diff = 0.00001
p = ti.field(dtype=ti.f32, shape=(N, N))  # 추가: 압력 필드
div = ti.field(dtype=ti.f32, shape=(N, N))  # 추가: 발산 필드

@ti.kernel  
def init_field():
    for i, j in ti.ndrange(N, N):
        density[i, j] = 0.0

@ti.kernel
def init_velocity():
    for i, j in ti.ndrange(N, N):
        if j >= 49*N//100 and j <= 51*N//100:
            vel[i, j] = ti.Vector([1.0, 0.0])
        else:
            vel[i, j] = ti.Vector([0.0, 0.0])

@ti.kernel
def add_source(density: ti.template(), source: ti.template(), dt: ti.f32):
    for i, j in density:
        density[i, j] += dt * source[i, j]

@ti.kernel
def diffuse(x: ti.template(), x0: ti.template(), diff: ti.f32, dt: ti.f32):
    a = dt * diff * N * N
    for k in range(20):
        for i, j in ti.ndrange(N, N):
            left = x[i-1, j] if i > 0 else x[i, j]
            right = x[i+1, j] if i < N-1 else x[i, j]
            down = x[i, j-1] if j > 0 else x[i, j]
            up = x[i, j+1] if j < N-1 else x[i, j]
            x[i, j] = (x0[i, j] + a * (left + right + down + up)) / (1 + 4 * a)

@ti.kernel
def advect(x: ti.template(), x0: ti.template(), vel: ti.template(), dt: ti.f32):
    dt0 = dt * N
    for i, j in ti.ndrange(N, N):
        xpos = i - dt0 * vel[i, j][0]
        ypos = j - dt0 * vel[i, j][1]
        if(xpos < 0.5): xpos = 0.5
        if(xpos > N + 0.5): xpos = N + 0.5
        if(ypos < 0.5): ypos = 0.5
        if(ypos > N + 0.5): ypos = N + 0.5

        i0 = int(xpos)
        i1 = i0 + 1
        j0 = int(ypos)
        j1 = j0 + 1

        s1 = xpos - i0
        s0 = 1 - s1
        t1 = ypos - j0
        t0 = 1 - t1

        x[i, j] = s0 * (t0 * x0[i0, j0] + t1 * x0[i0, j1]) + s1 * (t0 * x0[i1, j0] + t1 * x0[i1, j1])
        

def swap(a, b):
    tmp = a.to_numpy()
    a.copy_from(b)
    b.from_numpy(tmp)

@ti.func
def set_bnd_scalar(x: ti.template()):
    for i in range(N):
        x[i, 0] = 0.0        # 하단 경계
        x[i, N-1] = 0.0      # 상단 경계
        x[0, i] = 0.0        # 좌측 경계
        x[N-1, i] = 0.0      # 우측 경계

@ti.func
def set_bnd_vel(x: ti.template()):
    for i in range(N):
        x[i, 0][1] = -x[i, 1][1]      # 하단 경계: y-속도 반사
        x[i, N-1][1] = -x[i, N-2][1]  # 상단 경계: y-속도 반사
        x[0, i][0] = -x[1, i][0]      # 좌측 경계: x-속도 반사
        x[N-1, i][0] = -x[N-2, i][0]  # 우측 경계: x-속도 반사

@ti.kernel
def project(vel: ti.template(), p: ti.template(), div: ti.template()):
    # 발산 계산
    scale = 0.5 / N  # 그리드 크기에 맞춘 스케일링
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        div[i, j] = -0.5 * (
            vel[i+1, j][0] - vel[i-1, j][0] + 
            vel[i, j+1][1] - vel[i, j-1][1]
        ) * scale
        p[i, j] = 0.0  # 압력 초기화

    set_bnd_scalar(div)
    set_bnd_scalar(p)

    # Poisson 방정식 풀이 (Gauss-Seidel 반복)
    for k in range(20):  # 20회 반복으로 근사
        for i, j in ti.ndrange((1, N-1), (1, N-1)):
            p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4.0

    set_bnd_scalar(p)

    # 속도 보정
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        vel[i, j][0] -= (p[i+1, j] - p[i-1, j]) * (N * 0.5)
        vel[i, j][1] -= (p[i, j+1] - p[i, j-1]) * (N * 0.5)

    set_bnd_vel(vel)

init_field()
init_velocity()

def step():
    # add_source(density, source, dt)
    # swap(density, density_prev)
    # diffuse(density, density_prev, diff, dt)
    # swap(density, density_prev)
    # advect(density, density_prev, vel, dt)
    vel_step()
    dens_step()
    

def dens_step():
    add_source(density, source, dt)
    swap(density, density_prev)
    diffuse(density, density_prev, diff, dt)
    swap(density, density_prev)
    advect(density, density_prev, vel, dt)

def vel_step():
    swap(vel, vel_prev)
    diffuse(vel, vel_prev, diff, dt)
    project(vel, p, density_prev)
    swap(vel, vel_prev)
    advect(vel, vel_prev, vel, dt)
    project(vel, p, density_prev)

radius = 5
velsource = 2.0
gui = ti.GUI("Stable Fluid", res=(512, 512))
while gui.running:
    gui.get_event()
    source.fill(0.0)  # 매 프레임마다 소스 초기화
    if gui.is_pressed(ti.GUI.LMB):
        pos = gui.get_cursor_pos()
        grid_x = int(pos[0] * N)
        grid_y = int(pos[1] * N)
        for i in range(grid_x - radius, grid_x + radius + 1):
            for j in range(grid_y - radius, grid_y + radius + 1):
                if 0 <= i < N and 0 <= j < N:
                    if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:
                        source[i, j] = 100.0  # 소스에 값 추가
    if gui.is_pressed(ti.GUI.MMB):
        pos = gui.get_cursor_pos()
        grid_x = int(pos[0] * N)
        grid_y = int(pos[1] * N)
        for i in range(grid_x - radius, grid_x + radius + 1):
            for j in range(grid_y - radius, grid_y + radius + 1):
                if 0 <= i < N and 0 <= j < N:
                    if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:
                        vel[i, j][0] += dt*velsource  # x 방향 양의 속도 (오른쪽)
    
    # 오른쪽 마우스 버튼 (RMB): 왼쪽 방향 속도 추가
    if gui.is_pressed(ti.GUI.RMB):
        pos = gui.get_cursor_pos()
        grid_x = int(pos[0] * N)
        grid_y = int(pos[1] * N)
        for i in range(grid_x - radius, grid_x + radius + 1):
            for j in range(grid_y - radius, grid_y + radius + 1):
                if 0 <= i < N and 0 <= j < N:
                    if (i - grid_x)**2 + (j - grid_y)**2 <= radius**2:
                        vel[i, j][0] -= velsource  # x 방향 음의 속도 (왼쪽)

                    # 왼쪽 방향 속도
    
    step()
    density_np = density.to_numpy()
    gui.set_image(density_np)
    gui.show()
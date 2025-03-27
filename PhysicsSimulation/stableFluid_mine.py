import taichi as ti
import numpy as np  

ti.init(arch=ti.cpu)

N = 512
dt = 0.01
density = ti.field(dtype=ti.f32, shape=(N, N))
density_prev = ti.field(dtype=ti.f32, shape=(N, N))
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
vel_prev = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
source = ti.field(dtype=ti.f32, shape=(N, N))
diff = 0.0001

@ti.kernel
def init_field():
    for i, j in ti.ndrange(N, N):
        density[i, j] = 0.0

@ti.kernel
def init_velocity():
    for i, j in ti.ndrange(N, N):
        if j >= N//4 and j <= 3*N//4:
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

init_field()
init_velocity()

def step():
    add_source(density, source, dt)
    density_prev.copy_from(density)
    diffuse(density, density_prev, diff, dt)

radius = 5
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
    step()
    density_np = density.to_numpy()
    gui.set_image(density_np)
    gui.show()
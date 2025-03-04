import taichi as ti

ti.init(arch=ti.gpu)  # GPU 사용 (CPU 사용 시 ti.cpu)

# 시뮬레이션 파라미터 설정
N = 1000           # 입자 개수
dim = 2            # 2D 시뮬레이션
dt = 0.001         # 시간 간격
mass = 1.0         # 입자 질량
h = 0.04           # 스무딩 길이 (smoothing radius)
rho0 = 1000.0      # 기준 밀도
gravity = ti.Vector([0.0, -9.8])  # 중력 가속도

# 필드 선언
positions = ti.Vector.field(dim, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(dim, dtype=ti.f32, shape=N)
densities = ti.field(dtype=ti.f32, shape=N)
pressures = ti.field(dtype=ti.f32, shape=N)

@ti.kernel
def initialize():
    for i in range(N):
        positions[i] = ti.Vector([ti.random() * 0.5 + 0.25, ti.random() * 0.5 + 0.25])
        velocities[i] = ti.Vector([0.0, 0.0])

@ti.func
def poly6_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    factor = 315.0 / (64.0 * ti.math.pi * (h ** 9))
    return factor * (ti.max(0.0, (h * h - r * r))) ** 3

@ti.func
def spiky_gradient(r: ti.f32, h: ti.f32) -> ti.f32:
    factor = -45.0 / (ti.math.pi * (h ** 6))
    return factor * (ti.max(0.0, (h - r))) ** 2

@ti.kernel
def compute_density():
    for i in range(N):
        density = 0.0
        for j in range(N):
            r = (positions[i] - positions[j]).norm()
            density += mass * poly6_kernel(r, h)
        densities[i] = density

@ti.kernel
def compute_pressure():
    k = 2000.0
    for i in range(N):
        pressures[i] = k * (densities[i] - rho0)

@ti.kernel
def update():
    for i in range(N):
        pressure_force = ti.Vector([0.0, 0.0])
        viscosity_force = ti.Vector([0.0, 0.0])
        for j in range(N):
            if i != j:
                rij = positions[i] - positions[j]
                r = rij.norm()
                if r < h and r > 1e-5:
                    pressure_term = (pressures[i] + pressures[j]) / (2.0 * densities[j])
                    pressure_force += -mass * pressure_term * spiky_gradient(r, h) * (rij / r)
                    mu = 0.1
                    viscosity_force += mu * mass * (velocities[j] - velocities[i]) / densities[j]
        acceleration = pressure_force / densities[i] + viscosity_force + gravity
        velocities[i] += dt * acceleration
        positions[i] += dt * velocities[i]

@ti.kernel
def enforce_boundary():
    for i in range(N):
        if positions[i].x < 0.0:
            positions[i].x = 0.0
            velocities[i].x *= -0.5
        if positions[i].x > 1.0:
            positions[i].x = 1.0
            velocities[i].x *= -0.5
        if positions[i].y < 0.0:
            positions[i].y = 0.0
            velocities[i].y *= -0.5
        if positions[i].y > 1.0:
            positions[i].y = 1.0
            velocities[i].y *= -0.5

def main():
    initialize()
    gui = ti.GUI("SPH Fluid Simulation", res=(512, 512))
    while gui.running:
        for s in range(10):
            compute_density()
            compute_pressure()
            update()
            enforce_boundary()
        gui.circles(positions.to_numpy(), radius=2, color=0x068587)
        gui.show()

if __name__ == '__main__':
    main()

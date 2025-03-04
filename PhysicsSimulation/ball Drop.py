import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 10000  # 입자 수
dt = 0.001           # 타임 스텝
gravity = ti.Vector([0, -9.8])
radius = 0.02        # 입자 반지름

positions = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

@ti.kernel
def initialize():
    for i in range(n_particles):
        positions[i] = ti.Vector([ti.random() * 10, ti.random() * 10])
        velocities[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def update():
    for i in range(n_particles):
        velocities[i] += gravity * dt
        positions[i] += velocities[i] * dt

        # 경계 조건: 0~10 영역 내에서 반사 (입자 크기를 고려)
        if positions[i].x < radius:
            positions[i].x = radius
            velocities[i].x *= -0.5
        if positions[i].x > 10 - radius:
            positions[i].x = 10 - radius
            velocities[i].x *= -0.5
        if positions[i].y < radius:
            positions[i].y = radius
            velocities[i].y *= -0.5
        if positions[i].y > 10 - radius:
            positions[i].y = 10 - radius
            velocities[i].y *= -0.5

@ti.kernel
def collide():
    damping = 0.5  # 충돌 감쇠 계수 (0이면 완전 탄성, 1이면 비탄성)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dir = positions[i] - positions[j]
            dist_sq = dir.dot(dir)
            min_dist = 2 * radius
            if dist_sq < min_dist * min_dist:
                dist = ti.sqrt(dist_sq) + 1e-5  # 0으로 나누는 것을 방지
                overlap = min_dist - dist
                normal = dir / dist

                # 위치 보정: 겹친 만큼 각 입자를 반대 방향으로 이동
                correction = normal * (overlap * 0.5)
                positions[i] += correction
                positions[j] -= correction

                # 충돌에 따른 충격량 적용
                rel_vel = velocities[i] - velocities[j]
                vn = rel_vel.dot(normal)
                if vn < 0:  # 입자들이 서로 접근 중일 때만 적용
                    impulse = -(1 + damping) * vn * normal * 0.5  # 각 입자에 절반씩 적용
                    velocities[i] += impulse
                    velocities[j] -= impulse

def main():
    initialize()
    gui = ti.GUI("Stable 2D Collision Simulation", res=(600, 600))
    while gui.running:
        update()
        collide()  # 충돌 처리
        gui.clear(0x112F41)
        gui.circles(positions.to_numpy() / 10, radius=radius * 50)
        gui.show()

if __name__ == '__main__':
    main()

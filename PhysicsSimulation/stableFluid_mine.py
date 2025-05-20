import taichi as ti
import numpy as np  

ti.init(arch=ti.gpu)  # GPU 백엔드로 실행

N = 256
resolution = 512
dt = 0.1
density = ti.field(dtype=ti.f32, shape=(N, N))
density_prev = ti.field(dtype=ti.f32, shape=(N, N))
vel = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
vel_prev = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
source = ti.field(dtype=ti.f32, shape=(N, N))
diff = 0.00001
viscosity = 0.0001
p = ti.field(dtype=ti.f32, shape=(N, N))  # 압력 필드
p_prev = ti.field(dtype=ti.f32, shape=(N, N))  # Jacobi용 이전 압력 필드 
div = ti.field(dtype=ti.f32, shape=(N, N))  # 발산 필드
vort = ti.field(dtype=ti.f32, shape=(N, N))       # 스칼라 vorticity
vort_force = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))  # 힘 필드
epsilon = 3.0  # vorticity confinement의 강도
curl = 1.0

fire_img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))


@ti.kernel  
def init_field():
    for i, j in ti.ndrange(N, N):
        density[i, j] = 0.0
        if (i - N//2)**2 + (j - N//2)**2 < 100:
            density[i, j] = 0.0

@ti.kernel
def init_velocity():
    for i, j in ti.ndrange(N, N):
        if (i - N//2)**2 + (j - N//2)**2 < 50:
            vel[i, j] = ti.Vector([0.0, 0.0])
        else:
            vel[i, j] = ti.Vector([0.0, 0.0])

@ti.kernel
def add_source(x: ti.template(), s: ti.template(), dt: ti.f32):
    for i, j in x:
        x[i, j] += dt * s[i, j]

@ti.kernel
def diffuse(x: ti.template(), x0: ti.template(), diff: ti.f32, dt: ti.f32):
    a = dt * diff * N * N
    for k in range(20):
        for i, j in ti.ndrange(N, N):
            left = x0[i-1, j] if i > 0 else x0[i, j]
            right = x0[i+1, j] if i < N-1 else x0[i, j]
            down = x0[i, j-1] if j > 0 else x0[i, j]
            up = x0[i, j+1] if j < N-1 else x0[i, j]
            x[i, j] = (x0[i, j] + a * (left + right + down + up)) / (1 + 4 * a)

@ti.kernel
def advect(x: ti.template(), x0: ti.template(), vel: ti.template(), dt: ti.f32):
    dt0 = dt * N
    for i, j in ti.ndrange(N, N):
        xpos = min(max(i - dt0 * vel[i, j][0], 0.5), N - 0.5)
        ypos = min(max(j - dt0 * vel[i, j][1], 0.5), N - 0.5)
        i0, j0 = int(xpos), int(ypos)
        i1, j1 = i0 + 1, j0 + 1
        s1, t1 = xpos - i0, ypos - j0
        s0, t0 = 1 - s1, 1 - t1
        x[i, j] = (s0 * (t0 * x0[i0, j0] + t1 * x0[i0, j1]) +
                   s1 * (t0 * x0[i1, j0] + t1 * x0[i1, j1]))

@ti.kernel
def swap_fields(x: ti.template(), y: ti.template()):
    for i, j in ti.ndrange(N, N):
        x[i, j], y[i, j] = y[i, j], x[i, j]

@ti.func
def set_bnd_scalar(x: ti.template()):
    for i in range(N):
        x[i, 0] = 0.0
        x[i, N-1] = 0.0
        x[0, i] = 0.0
        x[N-1, i] = 0.0

@ti.func
def set_bnd_vel(x: ti.template()):
    for i in range(N):
        x[i, 0][1] = -x[i, 1][1]
        x[i, N-1][1] = -x[i, N-2][1]
        x[0, i][0] = -x[1, i][0]
        x[N-1, i][0] = -x[N-2, i][0]

# --- Jacobi 압력 프로젝션 ---
@ti.kernel
def compute_div(vel: ti.template(), div: ti.template()):
    scale = 0.5 / N
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        div[i, j] = -0.5 * (
            vel[i+1, j][0] - vel[i-1, j][0] +
            vel[i, j+1][1] - vel[i, j-1][1]
        ) * scale * curl
    set_bnd_scalar(div)

@ti.kernel
def zero_pressure(p: ti.template(), p_prev: ti.template()):
    for i, j in ti.ndrange(N, N):
        p[i, j] = 0.0
        p_prev[i, j] = 0.0
    set_bnd_scalar(p)
    set_bnd_scalar(p_prev)

@ti.kernel
def pressure_jacobi(p: ti.template(), p_prev: ti.template(), div: ti.template()):
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        p[i, j] = (div[i, j] + p_prev[i-1, j] + p_prev[i+1, j] +
                   p_prev[i, j-1] + p_prev[i, j+1]) * 0.25
    set_bnd_scalar(p)

@ti.kernel
def correct_vel(vel: ti.template(), p: ti.template()):
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        vel[i, j][0] -= (p[i+1, j] - p[i-1, j]) * (N * 0.5)
        vel[i, j][1] -= (p[i, j+1] - p[i, j-1]) * (N * 0.5)
    set_bnd_vel(vel)

@ti.kernel
def compute_vorticity():
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        dx = vel[i, j+1][0] - vel[i, j-1][0]
        dy = vel[i+1, j][1] - vel[i-1, j][1]
        vort[i, j] = 0.5 * (dx - dy) ## dy-dx가 수학적으로 맞음. 왜 0.5?

@ti.kernel
def apply_vorticity_confinement(epsilon: ti.f32):
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        # ∇|ω| 계산 (vorticity magnitude의 gradient)
        dw_dx = abs(vort[i+1, j]) - abs(vort[i-1, j])
        dw_dy = abs(vort[i, j+1]) - abs(vort[i, j-1])
        length = ti.sqrt(dw_dx**2 + dw_dy**2) + 1e-5
        Nv = ti.Vector([dw_dx, dw_dy]) / length

        # N x ω (2D에서 결과는 벡터)
        vort_dir = ti.Vector([0.0, 0.0])
        vort_dir[0] = -Nv[1] * vort[i, j]
        vort_dir[1] = Nv[0] * vort[i, j]

        vort_force[i, j] = epsilon * vort_dir

@ti.kernel
def add_force(vel: ti.template(), force: ti.template(), dt: ti.f32):
    for i, j in ti.ndrange(N, N):
        vel[i, j] += dt * force[i, j]

def fire_colormap(density):
    """
    density: 0~1 범위의 2D numpy array
    returns: (H, W, 3) RGB image, values in 0~1
    """
    h, w = density.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    # 불 느낌 색상 맵핑: 검정 → 빨강 → 주황 → 노랑 → 흰색
    for i in range(h):
        for j in range(w):
            d = density[i, j]
            if d < 0.2:
                img[i, j] = [d * 5, 0, 0]  # 검정 → 빨강
            elif d < 0.4:
                img[i, j] = [1.0, (d - 0.2) * 5, 0]  # 빨강 → 주황
            elif d < 0.7:
                img[i, j] = [1.0, 1.0, (d - 0.4) * 3.3]  # 주황 → 노랑
            else:
                val = min(1.0, (d - 0.7) * 3.3)
                img[i, j] = [1.0, 1.0, val]  # 노랑 → 거의 흰색

    return img

@ti.kernel
def compute_fire_color():
    for i, j in ti.ndrange(N, N):
        d = density[i, j]
        r, g, b = 0.0, 0.0, 0.0
        if d < 0.2:
            r = d * 5
        elif d < 0.4:
            r = 1.0
            g = (d - 0.2) * 5
        elif d < 0.7:
            r = 1.0
            g = 1.0
            b = (d - 0.4) * 3.3
        else:
            r = g = 1.0
            b = min(1.0, (d - 0.7) * 3.3)
        fire_img[i, j] = ti.Vector([r, g, b])


# step 함수들
init_field()
init_velocity()

##가우스 자이델은 gpu에서 병렬 스레드가 동시에 압력 필드를 업데이트 함. 
def project(vel, p, p_prev, div):
    compute_div(vel, div)
    zero_pressure(p, p_prev)
    for _ in range(50):  # 반복 횟수 조정 가능
        swap_fields(p, p_prev)
        pressure_jacobi(p, p_prev, div)
    correct_vel(vel, p)

##  without vorticity confinement
def vel_step():
    add_source(vel, vel_prev, dt)
    swap_fields(vel, vel_prev)
    diffuse(vel, vel_prev, viscosity, dt)
    project(vel, p, p_prev, div)
    swap_fields(vel, vel_prev)
    advect(vel, vel_prev, vel, dt)
    project(vel, p, p_prev, div)

##  with vorticity confinement
def vel_step_vc():
    add_source(vel, vel_prev, dt)
    swap_fields(vel, vel_prev)
    diffuse(vel, vel_prev, viscosity, dt)
    project(vel, p, p_prev, div)
    swap_fields(vel, vel_prev)
    advect(vel, vel_prev, vel, dt)
    compute_vorticity()
    apply_vorticity_confinement(epsilon)  # ε 값은 조정 가능
    add_force(vel, vort_force, dt)
    project(vel, p, p_prev, div)


def dens_step():
    add_source(density, source, dt)
    swap_fields(density, density_prev)
    diffuse(density, density_prev, diff, dt)
    swap_fields(density, density_prev)
    advect(density, density_prev, vel, dt)

# GUI 루프
radius = 5
velsource = 2.0

gui = ti.GUI("Stable Fluid (Jacobi)", res=(resolution, resolution))

diff_slider = gui.slider("diff", 0.0, 0.01)
diff_slider.value = diff

visc_slider = gui.slider("viscosity", 0.0, 0.01)
visc_slider.value = viscosity

eps_slider = gui.slider("epsilon", 0.0, 10.0)
eps_slider.value = epsilon

curl_slider = gui.slider("curl", 0.0, 5.0)
curl_slider.value = curl

use_vorticity_confinement = True  # Vorticity confinement 토글 플래그 추가
vort_button = gui.button("Toggle Vorticity Confinement")

while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == vort_button:
            use_vorticity_confinement = not use_vorticity_confinement
            
    source.fill(0.0)
    vel_prev.fill(ti.Vector([0.0, 0.0]))



    diff = diff_slider.value
    viscosity = visc_slider.value
    epsilon = eps_slider.value
    curl = curl_slider.value  # 발산 계산에 영향

    # if gui.button("Toggle Vorticity Confinement"):
    #     use_vorticity_confinement = not use_vorticity_confinement
    # gui.text(f"Vorticity Confinement: {'ON' if use_vorticity_confinement else 'OFF'}",pos=(0.01, 0.95), color=0xFFFFFF)

    if gui.is_pressed(ti.GUI.LMB) or gui.is_pressed(ti.GUI.RMB) or gui.is_pressed(ti.GUI.MMB):
        pos = gui.get_cursor_pos()
        gx, gy = int(pos[0] * N), int(pos[1] * N)
        for i in range(gx - radius, gx + radius + 1):
            for j in range(gy - radius, gy + radius + 1):
                if 0 <= i < N and 0 <= j < N and (i-gx)**2 + (j-gy)**2 <= radius**2:
                    if gui.is_pressed(ti.GUI.MMB): source[i, j] = 1.0
                    #if gui.is_pressed(ti.GUI.MMB): vel_prev[i, j] = ti.Vector([velsource, 0.0])
                    if gui.is_pressed(ti.GUI.RMB): vel_prev[i, j] = ti.Vector([0.0, velsource])

    
    if use_vorticity_confinement:
        vel_step_vc()
    else:
        vel_step()
    dens_step()
    # density_img = np.kron(density.to_numpy(), np.ones((resolution//N, resolution//N)))
    # gui.set_image(density_img)
    # density_np = density.to_numpy()
    # density_resized = np.kron(density_np, np.ones((resolution//N, resolution//N)))
    # density_rgb = fire_colormap(density_resized)
    # gui.set_image(density_rgb)

    compute_fire_color()  # 매 프레임 RGB 계산
    rgb_np = fire_img.to_numpy()
    rgb_resized = np.kron(rgb_np, np.ones((resolution//N, resolution//N, 1)))
    gui.set_image(rgb_resized)


    gui.show()

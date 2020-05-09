import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

dt = 0.01

xmin = 0
xmax = 3
ymin = -1
ymax = 1

fig = plt.figure() # initialise la figure
line, = plt.plot([],[])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

CORNER = np.array([xmin, ymin])
SPAN = np.array([(xmax - xmin), (ymax - ymin)])


CST_G = 0.0000001
# number of simulated steps
NUM_STEPS = 10000


class Cosmos:
    def __init__(self):
        self.body_list = []

    def add_body(self, new_body):
        self.body_list.append(new_body)

    def compute_forces(self, attracted):
        acc_force = np.zeros(2)
        for attractor in self.body_list:
            if attractor is attracted: continue
            acc_force += attractor.attractionTo(attracted)
        return acc_force

    def build_matrices(self):
        self.pos_matrix = np.stack([body.pos for body in self.body_list])
        self.speed_matrix = np.stack([body.current_speed for body in self.body_list])
        self.mass_matrix = np.array([body.mass for body in self.body_list])

    def update_pos_matrix(self, dt):
        shape_2d = (len(self.body_list), len(self.body_list))
        horyzontal_x = np.broadcast_to(self.pos_matrix[...,0], shape_2d)
        horyzontal_y = np.broadcast_to(self.pos_matrix[...,1], shape_2d)
        vertical_x = np.transpose(horyzontal_x)
        vertical_y = np.transpose(horyzontal_y)
        diff_x = horyzontal_x - vertical_x
        diff_y = horyzontal_y - vertical_y
        square_distance_matrix = diff_x * diff_x + diff_y * diff_y
        self.mass_matrix = self.mass_matrix.reshape((len(self.body_list), 1))
        mass_product_matrix = self.mass_matrix @ np.transpose(self.mass_matrix)
        # print("mass_product_matrix: ", mass_product_matrix)
        distance_matrix = np.sqrt(square_distance_matrix)
        # print("distance_matrix: ", distance_matrix)
        # valid for float64, uses to generate an identity matrix with infinity
        # as coefficient without multiplying 0 by infinity
        NEAR_INFINITY = 2**600
        distance_matrix = distance_matrix + NEAR_INFINITY * (NEAR_INFINITY * np.identity(len(self.body_list)))
        print("patched distance_matrix: ", distance_matrix)
        inv_distance_matrix = 1.0 / (distance_matrix)

        gravitational_force = inv_distance_matrix * inv_distance_matrix * mass_product_matrix * CST_G
        #print("gravitational_force: ", gravitational_force)
        unit_vector_x = diff_x / distance_matrix
        unit_vector_y = diff_y / distance_matrix
        #print("unit_vector_x: ", unit_vector_x)
        #print("unit_vector_y: ", unit_vector_y)
        reduce_matrix = np.array([1.0] * len(self.body_list)).reshape((len(self.body_list), 1))
        #print("reduce_matrix: ", reduce_matrix, reduce_matrix.dtype, reduce_matrix.shape)
        pre_mat_x = unit_vector_x * gravitational_force
        #print("pre_mat_x: ", pre_mat_x, pre_mat_x.dtype, pre_mat_x.shape)
        acc_x = (pre_mat_x.astype("float64")) @ reduce_matrix
        acc_y = (unit_vector_y * gravitational_force).astype("float64") @ reduce_matrix
        #print("acc_x: ", acc_x)
        #print("acc_y: ", acc_y)
        # speed matrix can be updated with dt and (acc_x, acc_y)
        acc_matrix = np.transpose(np.stack([acc_x, acc_y])).reshape((len(self.body_list), 2))
        #print("acc_matrix: ", acc_matrix)
        self.speed_matrix += dt * acc_matrix
        self.pos_matrix += dt * self.speed_matrix
        print("pos_matrix: ", self.pos_matrix)
        return self.pos_matrix

    def compute_acceleration(self, body):
        acceleration = self.compute_forces(body) / body.mass
        return acceleration

    def matrix_evolution(self, dt, time):
        pos_matrix = self.update_pos_matrix(dt)
        for index, body in enumerate(self.body_list):
            body.pos = pos_matrix[index]
        return map(lambda b: b.update_trajectory(time), self.body_list)

    def evolution(self, dt, time):
        acc_map = {}
        # NOTES: all the acceleration must be computed before any position is updated
        for body in self.body_list:
            acc_map[body] = self.compute_acceleration(body)
        for body in self.body_list:
            # speed update
            body.update_speed(dt, acc_map[body])
            # position update
            body.update_pos(dt)
        return map(lambda b: b.update_trajectory(time), self.body_list)


class Trajectory:
    def __init__(self, nb_points, index=0):
        self.points = np.zeros((2, nb_points))
        self.index = index

    def add_point(self, pt):
        self.points[0][self.index] = pt[0]
        self.points[1][self.index] = pt[1]
        self.index += 1

    @property
    def x(self):
        return self.points[0]
    @property
    def y(self):
        return self.points[1]

class Point:
    def __init__(self, pos):
        self.pos = pos

    def distance_to(pt0, pt1):
        return np.linalg.norm(pt0.pos - pt1.pos)
    def unit_vector_fromto(pt0, pt1):
        return (pt1.pos - pt0.pos) / pt0.distance_to(pt1)


class Body(Point):
    def __init__(self, init_pos, init_speed, nb_points=NUM_STEPS, mass=1, **plotargs):
        Point.__init__(self, init_pos)
        self.current_speed = init_speed
        self.trajectory = Trajectory(nb_points)
        self.plot, = plt.plot([], [], **plotargs)
        self.mass = mass

    def update_pos(self, dt):
        self.pos += dt * self.current_speed
    def update_speed(self, dt, acc):
        self.current_speed += dt * acc
    def update_trajectory(self, time):
        self.trajectory.add_point(self.pos)
        x = self.trajectory.x[time-10:time]
        y = self.trajectory.y[time-10:time]
        self.plot.set_data(x, y)
        return self.plot

    def attractionTo(attractor, attracted):
        value = CST_G * attracted.mass * attractor.mass / attractor.distance_to(attracted)**2
        unit_vector = attracted.unit_vector_fromto(attractor)
        return value * unit_vector


# listing bodies
body0 = Body(np.array([1.0, 0.5]), np.array([0, -0.5]), mass=2)
body1 = Body(np.array([2.5, -0.0]), np.array([0, +0.5]), mass=2)
body3 = Body(np.array([1.5, 0.]), np.array([0, +0.0]), mass=1000000)

def random_point():
    normalized_rand_pt = np.random.rand(2)
    return normalized_rand_pt * SPAN + CORNER

def random_body():
    init_pos = random_point()
    init_speed = np.random.rand(2) - np.array([0.5, 0.5])
    mass = 0.5 + random.random() * 10
    body = Body(
        init_pos,
        init_speed,
        mass=mass)
    return body

# initializing cosmos
universe = Cosmos()
universe.add_body(body0)
universe.add_body(body1)
universe.add_body(body3)
for i in range(10):
    break
    universe.add_body(random_body())

# adding massive central body
# universe.add_body(Body(CORNER + SPAN / 2, np.zeros(2), mass=1000000, linewidth=3))

universe.build_matrices()
print("pos_matrix: ", universe.pos_matrix)
print("speed_matrix: ", universe.speed_matrix)
print("mass_matrix: ", universe.mass_matrix)
#print("distance_matrix: ", universe.compute_distance_matrix())


def init():
    """ plot initialization """
    return [b.plot for b in universe.body_list]

def animate(i):
    """ plot i-th step """
    #return universe.evolution(dt, i)
    return universe.matrix_evolution(dt, i)

if True:
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=NUM_STEPS,
                                  blit=True, interval=20, repeat=False)

    plt.show()

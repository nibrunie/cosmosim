import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.01

xmin = 0
xmax = 3

fig = plt.figure() # initialise la figure
line, = plt.plot([],[])
plt.xlim(xmin, xmax)
plt.ylim(-1,1)


CST_G = 0.0001
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

    def compute_acceleration(self, body):
        return self.compute_forces(body) / body.mass

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
    def __init__(self, init_pos, init_speed, nb_points=NUM_STEPS, mass=1):
        Point.__init__(self, init_pos)
        self.current_speed = init_speed
        self.trajectory = Trajectory(nb_points)
        self.plot, = plt.plot([], [])
        self.mass = mass

    def update_pos(self, dt):
        self.pos += dt * self.current_speed
    def update_speed(self, dt, acc):
        self.current_speed += dt * acc
    def update_trajectory(self, time):
        self.trajectory.add_point(self.pos)
        x = self.trajectory.x[:time]
        y = self.trajectory.y[:time]
        self.plot.set_data(x, y)
        return self.plot

    def attractionTo(attractor, attracted):
        value = CST_G * attracted.mass * attractor.mass / attractor.distance_to(attracted)**2
        unit_vector = attracted.unit_vector_fromto(attractor)
        return value * unit_vector


# listing bodies
body0 = Body(np.array([1.0, 0.5]), np.array([0, -0.5]), mass=2)
body1 = Body(np.array([2.5, -0.0]), np.array([0, +0.5]), mass=2)
body3 = Body(np.array([1.5, 0.]), np.array([0, +0.0]), mass=10000)

# initializing cosmos
universe = Cosmos()
universe.add_body(body0)
universe.add_body(body1)
universe.add_body(body3)


def init():
    """ plot initialization """
    return [b.plot for b in universe.body_list]

def animate(i):
    """ plot i-th step """
    return universe.evolution(dt, i)

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=NUM_STEPS,
                              blit=True, interval=20, repeat=False)

plt.show()
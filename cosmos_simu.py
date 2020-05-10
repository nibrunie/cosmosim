import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import argparse
import math

dt = 10000.0#0.01



CST_G = 6.67408e-11 # m^3.kg^-1.s^-2

# number of simulated steps
NUM_STEPS = 10000


class Cosmos:
    def __init__(self, display_scale=np.array([1.0, 1.0]), gravitionnal_cst=CST_G):
        self.body_list = []
        # scale to be applied before display to reduce simulated
        # coords to window coords
        self.display_scale = display_scale
        # local cosmos gravitionnal constant
        self.gravitionnal_cst = gravitionnal_cst

    def add_body(self, new_body):
        self.body_list.append(new_body)

    def add_bodies(self, bodies):
        for body in bodies:
            self.add_body(body)

    def compute_forces(self, attracted):
        acc_force = np.zeros(2)
        for attractor in self.body_list:
            if attractor is attracted: continue
            acc_force += self.attractionTo(attractor, attracted)
        return acc_force

    def attractionTo(self, attractor, attracted):
        value = self.gravitionnal_cst * attracted.mass * attractor.mass / attractor.distance_to(attracted)**2
        unit_vector = attracted.unit_vector_fromto(attractor)
        return value * unit_vector

    def compile_matrices(self):
        # self.body_list update will not be taken into account between two
        # calls to compile_matrices
        self.NUM_BODIES = len(self.body_list)
        self.pos_matrix = np.stack([body.pos for body in self.body_list]).astype("float64")
        self.speed_matrix = np.stack([body.current_speed for body in self.body_list]).astype("float64")
        self.mass_matrix = np.array([body.mass for body in self.body_list], dtype="float64")
        self.shape_2d = (self.NUM_BODIES, self.NUM_BODIES)
        self.mass_matrix = self.mass_matrix.reshape((self.NUM_BODIES, 1))
        self.mass_product_matrix = self.mass_matrix @ np.transpose(self.mass_matrix)
        self.reduce_matrix = np.array([1.0] * self.NUM_BODIES, dtype="float64").reshape((self.NUM_BODIES, 1))
        self.inv_mass_matrix = np.float64(1.0) / self.mass_matrix

        self.weighted_g_matrix = (self.mass_product_matrix * self.gravitionnal_cst).astype("float64")
        # value valid for float64, uses to generate an identity matrix with infinity
        # as coefficient without multiplying 0 by infinity
        NEAR_INFINITY = np.float64(2**600)
        # diagnoal matrix with infinity as unique value (used to
        # set to 0 the diagnoal of 1.0 / distance_matrix)
        self.infinity_diag = NEAR_INFINITY * (NEAR_INFINITY * np.identity(self.NUM_BODIES))

    def update_pos_matrix(self, dt):
        horyzontal_x = np.broadcast_to(self.pos_matrix[...,0], self.shape_2d)
        horyzontal_y = np.broadcast_to(self.pos_matrix[...,1], self.shape_2d)
        vertical_x = np.transpose(horyzontal_x)
        vertical_y = np.transpose(horyzontal_y)
        diff_x = horyzontal_x - vertical_x
        diff_y = horyzontal_y - vertical_y
        square_distance_matrix = diff_x * diff_x + diff_y * diff_y
        distance_matrix = np.sqrt(square_distance_matrix)
        #square_distance_matrix = square_distance_matrix + self.infinity_diag
        distance_matrix = distance_matrix + self.infinity_diag
        inv_distance_matrix = np.float64(1.0) / (distance_matrix)

        gravitational_force = inv_distance_matrix * inv_distance_matrix * self.weighted_g_matrix
        unit_vector_x = diff_x * inv_distance_matrix
        unit_vector_y = diff_y * inv_distance_matrix
        acc_x = (unit_vector_x * gravitational_force) @ self.reduce_matrix
        acc_y = (unit_vector_y * gravitational_force) @ self.reduce_matrix

        acc_x *= self.inv_mass_matrix
        acc_y *= self.inv_mass_matrix
        # speed matrix can be updated with dt and (acc_x, acc_y)
        acc_matrix = np.transpose(np.stack([acc_x, acc_y])).reshape((self.NUM_BODIES, 2))
        self.speed_matrix += dt * acc_matrix
        self.pos_matrix += dt * self.speed_matrix
        return self.pos_matrix

    def compute_acceleration(self, body):
        acceleration = self.compute_forces(body) / body.mass
        return acceleration

    def matrix_evolution(self, dt, time, nb_steps=1):
        # simulate nb_steps before updating visual trajectory
        for step_id in range(nb_steps):
            pos_matrix = self.update_pos_matrix(dt)
        for index, body in enumerate(self.body_list):
            body.pos = pos_matrix[index]
        return map(lambda b: b.update_trajectory(time, self.display_scale), self.body_list)

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
        return map(lambda b: b.update_trajectory(time, self.display_scale), self.body_list)

    def animation(self, fig, nb_unit_steps=1):
        # matplotlib animation
        def init():
            """ plot initialization """
            return [b.plot for b in self.body_list]

        def animate(i):
            """ plot i-th step """
            #return universe.evolution(dt, i)
            return self.matrix_evolution(dt, i, nb_steps=nb_unit_steps)

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=NUM_STEPS,
                                      blit=True, interval=20, repeat=False)

        plt.show()


class Trajectory:
    """ visualized trajectory """
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
        # current 2D position
        self.pos = pos

    def distance_to(pt0, pt1):
        """ compute distance between Point pt0 and Point pt1 """
        return np.linalg.norm(pt0.pos - pt1.pos)
    def unit_vector_fromto(pt0, pt1):
        """ build a unit 2D vector from Point pt0 to Point pt1 """
        return (pt1.pos - pt0.pos) / pt0.distance_to(pt1)


class Body(Point):
    """ Astronomical body """
    def __init__(self, init_pos, init_speed, nb_points=NUM_STEPS, mass=1, plot_args={}):
        Point.__init__(self, init_pos)
        self.current_speed = init_speed
        self.trajectory = Trajectory(nb_points)
        self.plot, = plt.plot([], [], **plot_args)
        self.mass = mass

    def update_pos(self, dt):
        self.pos += dt * self.current_speed
    def update_speed(self, dt, acc):
        self.current_speed += dt * acc
    def update_trajectory(self, time, display_scale):
        scaled_point = self.pos * display_scale
        self.trajectory.add_point(scaled_point)
        x = self.trajectory.x[time-10:time]
        y = self.trajectory.y[time-10:time]
        self.plot.set_data(x, y)
        return self.plot



class Planet(Body):
    """ Solar system planet """
    def __init__(self, name, mass, radius, orbital_period_days=365.242, start_angle=0.0, start_tilt=0.0, speed_factor=1.0, plot_args=None):
        init_pos = np.array([radius * math.cos(start_angle), radius * math.sin(start_angle)], dtype="float64")
        speed_value = speed_factor * np.float64(2 * math.pi * radius / (orbital_period_days * 24.0 * 60.0 * 60.0))
        init_speed = speed_value * np.array([math.cos(math.pi/2.0 + start_angle + start_tilt), math.sin(math.pi/2.0 + start_angle + start_tilt)], dtype="float64")
        Body.__init__(self, init_pos, init_speed, mass=mass, plot_args=plot_args)
        self.name = name

def simulate_solar_system():
    # Simulating solar system using data from
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/
    xmin, xmax = -1, 2
    ymin, ymax = -1, 1

    fig = plt.figure() # initialise la figure
    line, = plt.plot([],[])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    SUN_MASS = 1.989e30 # kg
    MOON_MASS = 7.34767309e22 # kg
    ASTRO_UNIT = 1.49597e11 # m

    DISPLAY_SCALE = 0.5 / ASTRO_UNIT
    DISPLAY_SCALE_VECTOR = np.array([DISPLAY_SCALE, DISPLAY_SCALE], dtype="float64")

    sun = Body(np.array([0., 0.], dtype="float64"), np.array([0., 0.], dtype="float64"), mass=SUN_MASS, plot_args={"linewidth": 10, "color":"orange", "marker":"o"})
    
    VENUS_MASS = 4.87e24 # kg
    VENUS_RADIUS = 1.08e11 # meters
    VENUS_ORBITAL_PERIOD = 224.7 # days
    venus = Planet("venus", VENUS_MASS, VENUS_RADIUS, orbital_period_days=VENUS_ORBITAL_PERIOD, start_angle=math.pi/2.0, plot_args={"linewidth":3, "color":"grey"})


    EARTH_MASS = 5.972e24 # kg
    earth = Planet("earth", EARTH_MASS, ASTRO_UNIT, orbital_period_days=365.242, start_angle=math.pi, plot_args={"linewidth":3, "color":"blue"})

    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
    MARS_MASS = 6.39e23
    MARS_RADIUS = 2.2792e11 # semi-major axis
    MARS_YEAR = 687.973 # days, tropical orbit period
    mars = Planet("mars", MARS_MASS, MARS_RADIUS, orbital_period_days=MARS_YEAR, start_angle=0.0, plot_args={"linewidth":3, "color":"red"})

    JUPITER_MASS = 1.898e27
    JUPITER_ORBITAL_RADIUS = 7.786e11
    JUPITER_ORBITAL_PERIOD = 4331
    jupiter = Planet("jupiter", JUPITER_MASS, JUPITER_ORBITAL_RADIUS, orbital_period_days=JUPITER_ORBITAL_PERIOD, start_angle=0.0, plot_args={"linewidth":4, "color":"green"})

    SATURN_MASS = 5.68e26
    SATURN_ORBITAL_RADIUS = 1.4335e12
    SATURN_ORBITAL_PERIOD = 10747
    saturn = Planet("saturn", SATURN_MASS, SATURN_ORBITAL_RADIUS, orbital_period_days=SATURN_ORBITAL_PERIOD, start_angle=0.0, plot_args={"linewidth":4, "color":"purple"})


    solar_system = Cosmos(display_scale=DISPLAY_SCALE_VECTOR)
    solar_system.add_bodies([sun, venus, earth, mars, jupiter, saturn])

    # adding asteroids
    for i in range(200):
        asteroid_mass = 1e9 + random.random() * 5e9
        asteroid_orbital_radius = 6e11 + random.random() * 3e11
        asteroid = Planet("asteroid_%d" % i,
                          asteroid_mass,
                          asteroid_orbital_radius,
                          orbital_period_days=(2000),
                          speed_factor=0.0, #(0.75 + random.random() * 0.5),
                          start_angle=(math.pi * 2.0 * random.random()),
                          start_tilt=(random.random() * math.pi / 8 - math.pi / 16),
                          plot_args={"linewidth": 1, "color": "black"})
        solar_system.add_body(asteroid)

    solar_system.compile_matrices()

    solar_system.animation(fig, nb_unit_steps=10)


def demo():
    """ dummy demo """
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1

    fig = plt.figure() # initialise la figure
    line, = plt.plot([],[])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    CORNER = np.array([xmin, ymin])
    SPAN = np.array([(xmax - xmin), (ymax - ymin)])

    # listing bodies
    body0 = Body(np.array([1.0, 0.5]), np.array([0, -0.5]), mass=2)
    body1 = Body(np.array([2.5, -0.0]), np.array([0, +0.5]), mass=2)
    body3 = Body(np.array([1.5, 0.]), np.array([0, +0.0]), mass=1000000, linewidth=3)

    def random_point():
        normalized_rand_pt = np.random.rand(2)
        return normalized_rand_pt * SPAN + CORNER

    def random_body():
        init_pos = random_point()
        init_speed = 5.0 * (np.random.rand(2) - np.array([0.5, 0.5]))
        mass = 0.5 + random.random() * 1000
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
    for i in range(200):
        universe.add_body(random_body())

    # adding massive central body
    # universe.add_body(Body(CORNER + SPAN / 2, np.zeros(2), mass=1000000, linewidth=3))

    universe.compile_matrices()

    universe.animation(fig)



#for i in range(10000):
#    universe.matrix_evolution(dt, i)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cosmos simulator')
    parser.add_argument("--visualize", action="store_const", default=False,
                        const=True, help="enable graphical visualization")
    parser.add_argument("--num-steps", default=10000,
                        type=int, help="number of simulated steps")
    args = parser.parse_args()

    simulate_solar_system()

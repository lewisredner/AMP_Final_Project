# Lewis Redner Algorithmic Motion Planning Final Project

# import everything
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point, Polygon, LineString
from shapely import affinity
from shapely import wkt
import math
import matplotlib.pyplot as plt
import matplotlib
import time
from dijkstar import Graph, find_path


class RRT_Algorithm:

    def __init__(self, n, r, pgoal, eps, x_bounds, y_bounds, qstart, qgoal, speed, statics, dynamics, obsclass):
        self.qstart = qstart
        self.qgoal = qgoal
        self.n = n
        self.r = r
        self.pgoal = pgoal
        self.eps = eps
        self.statics = statics
        self.dynamics = dynamics
        self.xbounds = x_bounds
        self.ybounds = y_bounds
        self.speed = speed
        self.counter = 0
        self.obsclass = obsclass

        # initialise tree at q init
        self.TV = [self.qstart]
        self.TE = []

    #################################################################################################################################################
    # Goal Biased RRT algorithm
    def RRT_algo(self):
        # iterate while there is no solution
        for i in range(self.n):
            # generate a random sample with probability of pgoal of being goal
            # pdb.set_trace()
            qrand = self.random_sample()
            # find distance from each point to random sample, and take closest point in T
            qnear = self.find_min_distance_point(qrand)
            # make subpath from qnear to qrand
            path = self.make_path(qnear, qrand)
            # check if subpath is collision free
            if self.isSubpathCollisionFree(qnear, path):
                # take 1 step along path and add to tree as qnew
                qnew = [path.interpolate(self.r).xy[0][0], path.interpolate(self.r).xy[1][0]]
                # add edge (qnear, qnew) to T
                self.TE.append([qnear, qnew])
                self.TV.append(qnew)
                # check if the distance from qnew to qgoal is close to 0
                dist = math.sqrt((qnew[0] - self.qgoal[0]) ** 2 + (qnew[1] - self.qgoal[1]) ** 2)
                if len(self.TV) > 50000:
                    return 0, 0

                if dist <= self.eps:
                    # interpolate all of the vertices to make a series of points that the robot can follow
                    self.robot_positions = []
                    for edge in self.TE:
                        # take each edge and interpolate a distance equivalent 1/50th of the robot speed per second
                        diff_x = edge[1][0] - edge[0][0]
                        diff_y = edge[1][1] - edge[0][1]
                        dist_speed = self.speed * 1/10 # m/s * s = m
                        # now find number of points
                        num_points = int(diff_x/dist_speed)
                        # interpolate using these points
                        x_int = edge[0][0]
                        y_int = edge[0][1]
                        interpolateds = [x_int, y_int]
                        for i in range(num_points):
                            # append the robot positions
                            self.robot_positions.append(interpolateds)
                            # add the points in by moving the direction amount calculated in x and y
                            x_int = interpolateds[0]+dist_speed
                            y_int = interpolateds[1]+dist_speed
                            interpolateds = [x_int, y_int]


                        #self.robot_positions.append(np.linspace(edge[0],edge[1], num = self.speed/50,endpoint=True,retstep=True))

                    # if it is then return the solution path from root to q new
                    return self.TV, self.TE

    ########################################################################################################################
    # function to handle the motion of the robot through the workspace and dynamic obstacles
    def motion_updater(self):
        # get the current robot position
        current_robot_pos = self.robot_positions[0]
        # continue updating the motion of the things until we reach goal
        while current_robot_pos != self.qgoal:

            # get current position of robot
            current_robot_pos = self.robot_positions[0]

            # delete the old node as we don't need it anymore
            self.robot_positions.pop(0)
            viability = True
            # move the dynamic obstacles toward the robot
            if self.dynamics is not None:
                # update dynamic obstacle locations
                self.obsclass.propagate_dynamics(current_robot_pos)
                # check if the current motion plan is still possible (only needed in dynamic case)
                    # iterate over all points in robot positions and check if they are collision free
                temp = self.robot_positions.copy()
                for point in temp:
                    viability = self.obsclass.isPathViable(point)
                    # check if any point is no longer viable
                    if not viability:
                        # break
                        break

            # if viable, then update robot position
            if viability:
                # move the robot forward along its path
                current_robot_pos = self.robot_positions[0]
                # plot this
                self.plotting(self.nodes, self.total_cost)
            # if not, then make new RRT path
            else:
                # if no path exists, recalculate the motion path
                self.TV = [current_robot_pos.copy()]
                [self.TV, self.TE] = self.RRT_algo()

                # now now search for the path
                [nodes, cost] = self.graph_search()

                # now we can plot
                self.plotting(nodes, cost)

                # and run the motion updater again with our new motion plan
                self.motion_updater()


            # continue in loop until no solution reached 10 times, then return no solution



    #################################################################################################################################################
    # function to generate a random sample and return
    def random_sample(self):
        # generate random number between 0 and 100 and see if less than pgoal
        prob = random.uniform(0, 100) / 100
        if prob <= self.pgoal:
            return self.qgoal
        else:
            x_rand = random.uniform(self.xbounds[0], self.xbounds[1])
            y_rand = random.uniform(self.ybounds[0], self.ybounds[1])
            return [x_rand, y_rand]

    #################################################################################################################################################
    # calculate distance from each point in T to random sample
    def find_min_distance_point(self, random_point):
        # initialise distance array
        dist = []
        # iterate over all vertices in T
        for vertex in self.TV:
            # calculate distance between each point and save to array
            dist.append(math.sqrt((vertex[0] - random_point[0]) ** 2 + (vertex[1] - random_point[1]) ** 2))
            # find minimum and take same index to get closest point
        minpos = dist.index(min(dist))
        qnear = self.TV[minpos]
        return qnear

    #################################################################################################################################################
    # generates path from q near to qrand
    def make_path(self, qnear, qrand):
        # create points
        pointa = Point(qnear[0], qnear[1])
        pointb = Point(qrand[0], qrand[1])

        # create line string
        path = LineString([pointa, pointb])

        return path

    #################################################################################################################################################
    # check if path is collision free
    def isSubpathCollisionFree(self, qnear, path):
        # pdb.set_trace()
        # make path into a subpath by taking one step along the path
        start_point = [path.interpolate(0).xy[0][0], path.interpolate(0).xy[1][0]]
        step_point = [path.interpolate(self.r).xy[0][0], path.interpolate(self.r).xy[1][0]]
        # make the subpath
        subpath = LineString([Point(start_point[0], start_point[1]), Point([step_point[0], step_point[1]])])

        # set the is collision free to True until proven otherwise
        iscollisionfree = True

        # transfer shapely obstacles into shorthand
        if self.statics is not None:
            polys = self.statics
            # iterate over all obstacles and check for intersection
            for i in range(len(polys)):
                if polys[i].intersects(subpath):
                    iscollisionfree = False
                    break
        if self.dynamics is not None:
            polys = self.dynamics
            # repeat with dynamic obstacles
            for i in range(len(polys)):
                if polys[i].intersects(subpath):
                    iscollisionfree = False
                    break

        return iscollisionfree


    ###################################################################################################################
    # search the graph for the solution
    def graph_search(self):
        # set up the graph
        graph = Graph()
        # add edges
        for i in range(len(self.TE)):
            # pdb.set_trace()
            edge = self.TE[i]
            dist = math.sqrt((edge[0][0] - edge[1][0]) ** 2 + (edge[0][1] - edge[1][1]) ** 2)
            # pdb.set_trace()
            # find vertex from self.V that matches the edges
            vertex1 = self.TV.index(edge[0])
            vertex2 = self.TV.index(edge[1])
            # pdb.set_trace()
            graph.add_edge(vertex1, vertex2, dist)
        # pdb.set_trace()
        temp = find_path(graph, 0, len(self.TV) - 1)
        self.nodes = temp[0]
        self.total_cost = temp[3]
        return self.nodes, self.total_cost

    ###################################################################################################################
    # plot the path and the obstacles
    def plotting(self, key_nodes, dist):
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(111)
        # set limits
        plt.xlim(self.xbounds)
        plt.ylim(self.ybounds)

        # if there are static obstacles
        statics = self.statics
        if statics is not None:
            for i in range(len(statics)):
                # get coordinates of polygon vertices
                x, y = statics[i].exterior.coords.xy
                # get length of bounding box edges
                edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
                # get length of polygon as the longest edge of the bounding box
                width = max(edge_length)
                # get width of polygon as the shortest edge of the bounding box
                height = min(edge_length)
                rectangle = matplotlib.patches.Rectangle([min(x),min(y)], width, height, angle=0.0, fill=1)
                ax.add_patch(rectangle)
        # if there are dynamic obstacles
        dynamics = self.dynamics
        if dynamics is not None:
            for i in range(len(dynamics)):
                # get coordinates of polygon vertices
                x, y = dynamics[i].exterior.coords.xy
                # get length of bounding box edges
                edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
                # get length of polygon as the longest edge of the bounding box
                width = max(edge_length)
                # get width of polygon as the shortest edge of the bounding box
                height = min(edge_length)
                rectangle = matplotlib.patches.Rectangle([min(x), min(y)], width, height, angle=0.0, fill=1, facecolor = 'red')
                ax.add_patch(rectangle)

        # PLOT START, END AND CURRENT POINTS
        start = plt.Circle(self.qstart, 0.1, color='r')
        end = plt.Circle(self.qgoal, 0.1, color='g')
        curr = plt.Circle(self.robot_positions[0], 0.25, color='y')
        ax.add_artist(start)
        ax.add_artist(end)
        ax.add_artist(curr)

        # create the line for the path
        # import pdb; pdb.set_trace()
        for edge in self.TE:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'r')

        # add edges
        line = []
        for i in range(len(key_nodes) - 1):
            node1 = self.TV[key_nodes[i]]
            node2 = self.TV[key_nodes[i + 1]]
            # pdb.set_trace()
            line, = plt.plot([node1[0], node2[0]], [node1[1], node2[1]], 'b')

        # add vertices
        x, y = zip(*self.TV)
        points = plt.scatter(x, y, color='k')

        # line = plt.scatter(self.grid[:,0], self.grid[:,1])


        plt.legend((start, end, line, points), ('Start', 'End', 'Path', 'Sample Points'), loc=1)

        dist = round(total_cost, 2)
        title_boi = 'Goal Biased RRT Algorithm' + ', Distance: ' + str(dist)
        plt.title(title_boi)

        # show
        temp = "plan" + str(self.counter) + ".png"
        fig.savefig(temp)
        self.counter = self.counter + 1

        return

# class to create random and dynamic obstacles, and safe and dangerous areas in the workspace
class Obstacles:
    ###################################################################################
    # initialise everything
    def __init__(self, no_dynamics, dyn_speeds, dyn_size_x, dyn_size_y, no_statics, static_size_x, static_size_y):
        # inputs: number of dynamic obstacles, obstacle speeds, obstacle size, static obstacles, static obstacle size
        self.no_dynamic_obs = no_dynamics
        self.dyn_obstacle_speed = dyn_speeds
        self.dyn_size_x = dyn_size_x
        self.dyn_size_y = dyn_size_y
        self.no_static_obs = no_statics
        # bounds of the acceptable static size
        self.static_obstacle_x_bounds = static_size_x
        self.static_obstacle_y_bounds = static_size_y

        return

    ###################################################################################
    # create static obstacles
    def create_statics(self, xbounds, ybounds, start, goal):
        self.statics = []
        # iterate over desired number of obstacles
        for i in range(self.no_static_obs):
            # go until a valid obstacle is created
            while True:
                # generate random point in the workspace
                x_rand = random.uniform(xbounds[0], xbounds[1])
                y_rand = random.uniform(ybounds[0], ybounds[1])

                # choose a static size from the bounds provided
                x_size = random.uniform(self.static_obstacle_x_bounds[0],self.static_obstacle_x_bounds[1])
                y_size = random.uniform(self.static_obstacle_y_bounds[0], self.static_obstacle_y_bounds[1])

                # find the vertices (lower left, upper left, upper right, lower right)
                v1 = [x_rand - x_size / 2, y_rand - y_size / 2]
                v2 = [x_rand - x_size / 2, y_rand + y_size / 2]
                v3 = [x_rand + x_size / 2, y_rand + y_size / 2]
                v4 = [x_rand + x_size / 2, y_rand - y_size / 2]

                # create the shapely object
                static_obs = Polygon([v1,v2,v3,v4])

                # check if goal in there
                if not static_obs.intersects(Point(start)) and not static_obs.intersects(Point(goal)):
                    self.statics.append(static_obs)
                    break

        return self.statics

    ###################################################################################
    # create safe and dangerous space
    def create_spaces(self):
        # iterate over number of safe spaces

        # select random point

        # create shapely object using area size around that point

        # iterate over number of dangerous spaces

        # repeat procedure above, make sure there is no overlap with a safe space


        return

    ###################################################################################
    # create the dynamic objects
    def create_dynamics(self, xbounds, ybounds, start, goal):
        # same procedure as creating the static obstacle, just make sure it doesn't intersect an existing static
        self.dynamics = []
        # iterate over desired number of obstacles
        for i in range(self.no_dynamic_obs):
            # go until a valid obstacle is created
            while True:
                # generate random point in the workspace
                x_rand = random.uniform(xbounds[0], xbounds[1])
                y_rand = random.uniform(ybounds[0], ybounds[1])

                # choose a static size from the bounds provided
                x_size = random.uniform(self.dyn_size_x[0], self.dyn_size_x[1])
                y_size = random.uniform(self.dyn_size_y[0], self.dyn_size_y[1])

                # find the vertices (lower left, upper left, upper right, lower right)
                v1 = [x_rand - x_size / 2, y_rand - y_size / 2]
                v2 = [x_rand - x_size / 2, y_rand + y_size / 2]
                v3 = [x_rand + x_size / 2, y_rand + y_size / 2]
                v4 = [x_rand + x_size / 2, y_rand - y_size / 2]

                # create the shapely object
                dynamic_obs = Polygon([v1, v2, v3, v4])

                # check if the obstacle intersects goal
                if not dynamic_obs.intersects(Point(start)) and not dynamic_obs.intersects(Point(goal)):
                    # check if the dynamic obstacle intersects the static
                    for static in self.statics:
                        # if there is an intersection, then continue
                        if dynamic_obs.intersects(static):
                            continue
                    # append the dynamic obstacle to the list
                    self.dynamics.append(dynamic_obs)
                    break

        return self.dynamics


    ###################################################################################
    # propagate the dynamic objects
    def propagate_dynamics(self, robot_pos):
        # inputs: array of tuples for dynamic obstacle position, and robot position as [x,y]
        # keep track of which is being updated
        index = 0
        for dynamic in self.dynamics:
            # convert obstacle and robot positions into Shapely points
            r_pos = Point(robot_pos)
            object_centre = dynamic.centroid


            # create a LineString to connect the two together
            direct_path = LineString([object_centre, r_pos])

            # go along that path for 1 timestep (1/10s) at 1/2 the speed of the robot
            new_centre_point = direct_path.interpolate(self.dyn_obstacle_speed/20).xy

            # recreate the obstacle around the point
            travel_distx = new_centre_point[0][0] - object_centre.xy[0][0]
            travel_disty = new_centre_point[1][0] - object_centre.xy[1][0]

            # translate the object
            new_loc = affinity.translate(dynamic, travel_distx, travel_disty)

            # update the dynamic obstacle
            self.dynamics[index] = new_loc
            # update the index
            index = index + 1
        return

    ###################################################################################
    # create intersection checker
    def isPathViable(self, point):
        pt = Point(point)
        iscollisionfree = True
        # transfer shapely obstacles into shorthand
        if self.statics is not None:
            polys = self.statics
            # iterate over all obstacles and check for intersection
            for i in range(len(polys)):
                if polys[i].intersects(pt):
                    iscollisionfree = False
                    break
        if self.dynamics is not None:
            polys = self.dynamics
            # repeat with dynamic obstacles
            for i in range(len(polys)):
                if polys[i].intersects(pt):
                    iscollisionfree = False
                    break

        return iscollisionfree


# MAIN SCRIPT
no_dynamics = 1
dyn_speeds = 1
dyn_size_x = [1,1]
dyn_size_y = [1,1]
no_statics = 4
static_size_x = [2,5]
static_size_y = [2,5]
obs = Obstacles(no_dynamics, dyn_speeds, dyn_size_x, dyn_size_y, no_statics, static_size_x, static_size_y)
start = [0,0]
goal = [10,10]
xbounds = [0,20]
ybounds = [0,15]
static_obs = obs.create_statics(xbounds, ybounds, start, goal)
dynamic_obs = obs.create_dynamics(xbounds, ybounds, start, goal)
# number of samples and radius of connectivity
n = 500
r = 0.5
# probability of reaching goal
p_goal = 0.05
# set eps
eps = 0.1
# set the robot and obstacle speeds [m/s]
speed = 0.5



# instantiate the solving class
rrt = RRT_Algorithm(n, r, p_goal, eps, xbounds, ybounds, start, goal, speed, static_obs, dynamic_obs, obs)
# solve the current motion problem
[V, E] = rrt.RRT_algo()
# search the graph
[nodes, total_cost] = rrt.graph_search()
# plot the current solution
rrt.plotting(nodes, total_cost)
# update the motion by propagating the robot position
rrt.motion_updater()


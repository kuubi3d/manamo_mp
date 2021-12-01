
import sys
from IPython import get_ipython
import time
import argparse

import msgpack
from enum import Enum, auto

import numpy as np
from sympy import interpolate

# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
#from udacidrone.drone import set_home_position

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import networkx as nx

from scipy import interpolate

import matplotlib.pyplot as plt
from networkx import Graph
import graphviz
import re

from copy import deepcopy

get_ipython().run_line_magic('matplotlib', 'inline')
plt.switch_backend('Qt5agg')

plt.rcParams['figure.figsize'] = 12, 12



class RRT:

    x_goal = (30, 750)
    rrt_goal = ()
    num_vertices = 1600
    dt = 18
    x_init = (20, 150)
    path = [(20, 30), (40, 50)]
     
    path_cost = 0
    g = graphviz.Digraph('RRT Path', format = 'svg', filename='hello.gv')

    def __init__(self, x_init):
        # A tree is a special case of a graph with
        # directed edges and only one path to any vertex.
        self.tree = nx.DiGraph()
        self.tree.add_node(x_init)

        self.path_tree = nx.DiGraph()
        self.path_tree.add_node(x_init)

        
    def add_vertex(self, x_new):
        self.tree.add_node(tuple(RRT.x_init))
    
    def add_edge(self, x_near, x_new, u):
        self.tree.add_edge(tuple(x_near), tuple(x_new), orientation=u)
        
    @property
    def vertices(self):
        return self.tree.nodes()
    
    @property
    def edges(self):
        return self.tree.edges()

    
    def add_rrt_vertex(self, x_new):
        self.path_tree.add_node(tuple(RRT.x_init))
    
    def add_rrt_edge(self, x_near, x_new, u):
        self.path_tree.add_edge(tuple(x_near), tuple(x_new), orientation=u)

    
    def rrt_vertices(self):
        return self.path_tree.nodes()
   
    @property
    def rrt_edges(self):
        return self.path_tree.edges()

    @property
    def parent(self, x_new):
        return self.tree.predecessors(x_new)

    
    def path_nodes(self):
        return list(self.path_tree.nodes)

    def get_parent(self, x_new):
        return self.tree.predecessors(x_new)


    def create_grid(self, data, drone_altitude, safety_distance):
        """
        Returns a grid representation of a 2D configuration space
        based on given obstacle data, drone altitude and safety distance
        arguments.
        """
    
        # minimum and maximum north coordinates
        north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
        north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

        # minimum and maximum east coordinates
        east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
        east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))



        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
        north_size = int(np.ceil(north_max - north_min))
        east_size = int(np.ceil(east_max - east_min))

        print("north min, max, and size", north_max, north_min, north_size)
        print("east min, max, and size", east_max, east_min, east_size)
        # Initialize an empty grid
        grid = np.zeros((north_size, east_size))

        # Populate the grid with obstacles
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            if alt + d_alt + safety_distance > drone_altitude:
                obstacle = [
                    int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                    int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                    int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                    int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
                ]
                grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
        
        print('INFO', obstacle, drone_altitude, safety_distance)
        print(grid, int(north_min), int(east_min))        
    

       
        return grid, int(north_min), int(east_min)
    
    def sample_state(self, grid):
        x = np.random.uniform(0, grid.shape[0])
        y = np.random.uniform(0, grid.shape[1])
        return (x, y)


    # ### Nearest Neighbors
    # 
    # A critical part of the RRT procedure is finding the closest vertex to the sampled random point. This the most computationally intensive part so be mindful of that. Depending on the number of vertices a naive implementation will run into trouble quickly.


    def nearest_neighbor(self, x_rand, rrt):
        
               
        #wp_radius = np.linalg.norm(x_goal)
        #print ('waypoint radius', wp_radius)
    
        closest_dist = 100000
        closest_vertex = None
        x_rand = np.array(x_rand)

        print ("x_rand", x_rand)       
        

        for v in rrt.vertices:
            d = np.linalg.norm(x_rand - np.array(v[:2]))
            if d < closest_dist:
                closest_dist = d
                closest_vertex = v
            
        return closest_vertex


    # ### Selecting Inputs
    # 
    # Select input which moves `x_near` closer to `x_rand`. This should return the angle or orientation of the vehicle.


    def select_input(self, x_rand, x_near):
        return np.arctan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])


    # ### New State
    # 
    # 

    # The new vertex `x_new` is calculated by travelling from the current vertex `x_near` with a orientation `u` for time `dt`.


    def new_state(self, x_near, u, dt):
        nx = x_near[0] + np.cos(u)*dt
        ny = x_near[1] + np.sin(u)*dt
        return (nx, ny)


    # ### Putting It All Together
    # 
    # Awesome! Now we'll put everything together and generate an RRT.

    
    def generate_RRT(self, grid, x_init, x_goal, num_vertices, dt):
       
        
        #x_goal = (30, 750)
        
        num_vertices = 1600
        dt = 18
        #x_init = (20, 150)
        path = [(20, 30), (40, 50)]

        print ('Planning RRT path. It may take a few seconds...')
        rrt = RRT(x_init)
        rrt_path = RRT(x_init)
        #plt.imshow(grid, cmap='Greys', origin='lower')
        sys.exit
        print("grid shape", grid.shape, grid)


        for _ in range(num_vertices):


            x_rand = RRT.sample_state(self, grid)
            # sample states until a free state is found
            while grid[int(x_rand[0]), int(x_rand[1])] == 1:
                x_rand = RRT.sample_state(self, grid)
                                  
            x_near = RRT.nearest_neighbor(self, x_rand, rrt)
            u = RRT.select_input(self, x_rand, x_near)
            x_new = RRT.new_state(self, x_near, u, dt)
            
            norm_g = np.array(x_goal)
            norm_n = np.array(x_near)
            
           
            if np.linalg.norm(norm_g - norm_n) < 200:

                print ("Goal Found.")
                rrt.add_edge(x_near, x_new, u)

                # Now let's plot the generated RRT.

               
                plt.imshow(grid, cmap='Greys', origin='lower')
                plt.plot(RRT.x_init[1], RRT.x_init[0], 'ro')
                plt.plot(RRT.x_goal[1], RRT.x_goal[0], 'ro')
            
                for (v1, v2) in rrt.edges:
                    plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'y-')
                
                plt.show(block=True)
               
                current_node = x_new

                for _ in range(num_vertices):

                    parent = list(rrt.get_parent(current_node))
                    current_node = (int(current_node[0]), int(current_node[1]))
                    parent_node = tuple(round(int(p1)) for p1 in parent[0])
                    
                    print("current_node", current_node)
                    print("parent node", parent_node)

                    rrt_path.add_rrt_edge(current_node, parent_node, u)
                    
                    current_node = tuple(parent[0])
                    print("new parent", current_node)
                    
                    if parent_node == x_init:

                        print("Path Mapped")
                        
                        #added to shift waypoints on the simulator if needed.
                        shft_x, shft_y = [0,0] 
                        #print ("Shift", shft_x, shft_y)
                        #RRT.wp_nodes = list((a-shft_x, b-shft_y) for a, b in rrt_path.path_tree.nodes)


                        print (rrt_path.path_tree.edges, "\n")
                        print (rrt_path.path_tree.nodes, "\n")

                       
                        path_list = [list(x) for x in (rrt_path.path_tree.nodes)]
                        print ("path_list", path_list,"\n")
                        
                        s_path = RRT.smooth(path_list)
                        #print("smoothed", rrt_path)
                        print("smooth_path", s_path)
                        #print("networkx_path", rrt_path.path_tree.edges )



                        plt.imshow(grid, cmap='Greys', origin='lower')
                        plt.plot(RRT.x_init[1], RRT.x_init[0], 'ro')
                        plt.plot(RRT.x_goal[1], RRT.x_goal[0], 'ro')

                        #for (v1, v2) in RRT.wp_nodes:
                        for (v1, v2) in s_path:

                            plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'y-')
                        
                        plt.show(block=True)

                        return rrt_path


                        

            elif grid[int(x_new[0]), int(x_new[1])] == 0:
                # the orientation `u` will be added as metadata to
                # the edge
                rrt.add_edge(x_near, x_new, u)
               

        print("RRT Path Mapped")    
        return rrt_path   
            
        #States
     
    # Assume all actions cost the same.

    def heuristic(position, goal_position):
        return np.linalg.norm(np.array(position) - np.array(goal_position))

    def smooth(s_path, weight_data=.1, weight_smooth=.9, tolerance=0.0001):
        """
        Creates a smooth path for a n-dimensional series of coordinates.

        Arguments:
            path: List containing coordinates of a path
            weight_data: Float, how much weight to update the data (alpha)
            weight_smooth: Float, how much weight to smooth the coordinates (beta).
            tolerance: Float, how much change per iteration is necessary to keep iterating.

        Output:
            new: List containing smoothed coordinates.
        """
        
        new = deepcopy(s_path)
        rrt_path = RRT(RRT.x_init)
        dims = len(s_path[0])
        change = tolerance
        smooth = []
        while change >= tolerance:
            change = 0.0
            
            for i in range(1, len(new) - 1):
                for j in range(dims):

                    x_i = s_path[i][j]
                    y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]
                    #print("y_i, y_prev, y_next", y_i, y_prev, y_next)

                    y_i_saved = y_i
                    
                    print("change, weight data", change, weight_data)

                    y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                    print("y_i", y_i)
                    new[i][j] = round(y_i, ndigits=4)
                    #print("y_1, y_2", "y_3", y_1, y_2, "\n",y_3)
                    
                    change += abs(y_i - y_i_saved)
                    print("change", change)
                    print("new", new, "\n")

            new = list(tuple(x) for x in new)
            x = 0
            RRT.s_path = []
            for i in range(0, len(new)-1):
                print ("len", len(new))

                rrt_path.add_rrt_vertex(new[i])
                
                s_path[i] = ((new[i], new[i+1]))
            
            s_path.pop()
           
            
            
            """
            s_path1 = [((20, 150), (118.0, 429.5)), ((118.0, 429.5), (176.0, 569.75)), ((176.0, 569.75), (210.5, 647.375)),
                       ((210.5, 647.375), (236.25, 683.6875)), ((236.25, 683.6875), (254.625, 708.8438)), ((254.625, 708.8438), (271.8125, 716.9219)),
                       ((271.8125, 716.9219), (288.9062, 724.461)), ((288.9062, 724.461), (305.9531, 726.2305)), ((305.9531, 726.2305), (323.4765, 725.1153)), 
                       ((305.9531, 726.2305), (323.4765, 725.1153)), ((323.4765, 725.1153), (341.2382, 723.5576)), ((341.2382, 723.5576), (358.6191, 720.2788)),
                       ((358.6191, 720.2788), (372.8096, 711.1394)), ((372.8096, 711.1394), (387.9048, 703.0697)), ((387.9048, 703.0697), (398.9524, 690.5349)),
                       ((398.9524, 690.5349), (396.9762, 679.2675)), ((396.9762, 679.2675), (396.9881, 665.1337)), ((396.9881, 665.1337), (405.4941, 654.5668)),
                       ((405.4941, 654.5668), (413.2471, 640.7834)), ((413.2471, 640.7834), (425.1236, 629.8917)), ((425.1236, 629.8917), (432.5618, 615.4459)),
                       ((432.5618, 615.4459), (444.2809, 604.723)), ((444.2809, 604.723), (453.6404, 590.8615)), ((453.6404, 590.8615), (454.8202, 575.4307)),
                       ((454.8202, 575.4307), (455.9101, 558.7153)), ((455.9101, 558.7153), (459.4551, 542.3576)), ((459.4551, 542.3576), (465.2276, 525.6788)),
                       ((465.2276, 525.6788), (474.1138, 510.8394)), ((474.1138, 510.8394), (478.5569, 494.4197)), ((478.5569, 494.4197), (485.2785, 478.7098)),
                       ((485.2785, 478.7098), (482.1393, 464.8549)), ((482.1393, 464.8549), (484.5697, 449.4275)), ((484.5697, 449.4275), (479.7849, 435.2138)),
                       ((479.7849, 435.2138), (472.8924, 420.1069)), ((472.8924, 420.1069), (470.9462, 403.5534)), ((470.9462, 403.5534), (467.9731, 386.7767)),
                       ((467.9731, 386.7767), (457.4865, 375.8884)), ((457.4865, 375.8884), (449.2432, 361.9442)), ((449.2432, 361.9442), (442.1216, 346.4721)),
                       ((442.1216, 346.4721), (430.0608, 341.736)), ((430.0608, 341.736), (420.0304, 330.868)), ((420.0304, 330.868), (409.5152, 318.434)),
                       ((409.5152, 318.434), (395.7576, 309.717)), ((395.7576, 309.717), (379.8788, 304.3585)), ((379.8788, 304.3585), (371.4394, 310.6793)),
                       ((371.4394, 310.6793), (358.7197, 316.3397)), ((358.7197, 316.3397), (343.3598, 320.6698)), ((343.3598, 320.6698), (330.6799, 330.3349)),
                       ((330.6799, 330.3349), (315.8399, 331.6675)), ((315.8399, 331.6675), (300.4199, 328.8338)), ((300.4199, 328.8338), (284.2099, 330.9169)),
                       ((284.2099, 330.9169), (267.605, 335.4584)), ((267.605, 335.4584), (250.3025, 338.7292)), ((250.3025, 338.7292), (233.1513, 338.3646)),
                       ((233.1513, 338.3646), (215.5756, 339.6823)), ((215.5756, 339.6823), (201.2878, 332.8411)), ((201.2878, 332.8411), (191.1439, 321.4205)),
                       ((191.1439, 321.4205), (181.072, 307.7102)), ((181.072, 307.7102), (177.536, 291.8551)), ((177.536, 291.8551), (170.768, 276.9275)),
                       ((170.768, 276.9275), (158.384, 269.9638)), ((158.384, 269.9638), (144.692, 261.9819)), ((144.692, 261.9819), (128.846, 256.4909)),
                       ((128.846, 256.4909), (121.923, 244.7455)), ((121.923, 244.7455), (112.9615, 231.3727)), ((112.9615, 231.3727), (103.4808, 217.1864)),
                       ((103.4808, 217.1864), (93.7404, 203.0932)), ((93.7404, 203.0932), (80.3702, 193.5466)), ((80.3702, 193.5466), (65.1851, 191.2733)),
                       ((65.1851, 191.2733), (49.0926, 186.6367)), ((49.0926, 186.6367), (39.5463, 175.3184)), ((39.5463, 175.3184), (30, 164))] #[30, 164]]

            #"""  
            
            return s_path

        
class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)
    
    # ~ :::: from darienmt planning_util

    def read_home(filename):
        """
        Reads home (lat, lon) from the first line of the `file`.
        """
        with open(filename) as f:
            first_line = f.readline()
        match = re.match(r'^lat0 (.*), lon0 (.*)$', first_line)
        if match:
            lat = match.group(1)
            lon = match.group(2)
        return np.fromstring(f'{lat},{lon}', dtype='float64', sep=',')

    #  :::: ~


    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 20
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # ~ :::: from darienmt

        colliders_file = 'colliders.csv'
        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = MotionPlanning.read_home(colliders_file)
        print(f'Home lat : {lat0}, lon : {lon0}')
        # # # TODO: set home position to (lat0, lon0, 0)
        self.set_home_position(lon0, lat0, 0)
        self.set_home_as_current_position()
        # TODO: retrieve current global position
        local_north, local_east, local_down = global_to_local(self.global_position, self.global_home)
        print(f'Local => north : {local_north}, east : {local_east}, down : {local_down}')
        self._update_global_home

        
        # TODO: convert to current local position using global_to_local()
        
        #  :::: ~
       
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        #grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        grid, north_offset, east_offset = RRT.create_grid(self, data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)
        #grid_start = (20, 150)
        # TODO: convert start position to current position rather than map center
        
        # Set goal as some arbitrary position on the grid
        grid_goal = (-north_offset + 10, -east_offset + 10)
        #grid_goal = (30, 750)
       
        # TODO: adapt to set goal as latitude / longitude position and convert

        # Run A* to find a path from start to goal
        
        self.local_position_callback
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        
        
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        
        
        rrt_path = RRT.generate_RRT(self, grid, RRT.x_init, RRT.x_goal, RRT.num_vertices, RRT.dt)
       
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        #self.send_waypoints()

        print("rrt_path for waypoints", rrt_path.path_tree.nodes)
        waypoints = [[r[0] + north_offset, r[1] + east_offset, TARGET_ALTITUDE, 0] for r in RRT.s_path ]
        waypoints = list(reversed(waypoints))
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        
        print("waypoints", waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=240)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()

    

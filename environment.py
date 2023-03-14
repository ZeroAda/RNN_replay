import numpy as np
import random
import tensorflow as tf
import itertools
class gameOb():
    def __init__(self ,coordinates ,size ,color ,reward ,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name

class mazeworld():
    def __init__(self ,size):
        self.sizeX = size
        self.sizeY = size
        self.num_actions = 4
        self.state_dim = 16  # one hot state encoding
        self.objects = []
        self.timestep = 0
        self.time = 0
        self.bg = np.zeros([size ,size])

        a = self.reset()




    def getFeatures(self):
        return np.array([self.objects[0].x ,self.objects[0].y]) / float(self.sizeX)

    def findNeighbor(self ,state):
        neighbor = list()
        # up
        neighbor.append([(state[0 ] -1 ) %self.sizeX, state[1]])
        # down
        neighbor.append([(state[0 ] +1 ) %self.sizeX, state[1]])
        # left
        neighbor.append([state[0], (state[1 ] -1 ) %self.sizeY])
        # right
        neighbor.append([state[0], (state[1 ] +1 ) %self.sizeY])

        return neighbor

    def walkMaze(self, state, visited, wall):
        visited.append(state)
        # add neightbors of s
        neighbors = self.findNeighbor(state)
        # randomize neighbor
        random.shuffle(neighbors)
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            # if a state not visited
            if neighbor not in visited:
                # remove wall

                state_ind = state[0] * 4 + state[1]
                neighbor_ind = neighbor[0] * 4 + neighbor[1]
                wall[state_ind, neighbor_ind] = 1
                wall[neighbor_ind, state_ind] = 1

                # walk maze
                wall, visited = self.walkMaze(neighbor, visited, wall)

        return wall ,visited

    def initWall(self):
        visited = []
        # walls everywhere
        wall = np.ones([self.sizeX**2 ,self.sizeX**2])
        # initialize wall
        for i in range(self.sizeX**2):
            state = [ i//4, i% 4]
            neighbors = self.findNeighbor(state)
            nb = np.zeros(self.sizeX)
            # walls everywhere
            for neighbor in neighbors:
                neighbor_ind = neighbor[0] * 4 + neighbor[1]
                wall[i, neighbor_ind] = 0
                wall[neighbor_ind, i] = 0

        state = [np.random.randint(0, self.sizeX), np.random.randint(0, self.sizeY)]
        wall, visited = self.walkMaze(state, visited, wall)
        # remove some wall randomly
        barrier = wall.copy()
        where_0 = np.where(wall == 0)
        where_1 = np.where(wall == 1)

        barrier[where_0] = 1
        barrier[where_1] = 0
        src, dst = np.nonzero(barrier)
        i = 0
        while i < 10:
            s = np.random.randint(0, len(src))
            p = src[s]
            q = dst[s]
            if p == q:
                continue
            wall[p, q] = 1
            wall[q, p] = 1
            i += 1

        self.wall = wall

    def reset(self):
        self.objects = []

        # randomly set a hero position
        self.hero = gameOb(self.newPosition(0), 1, [0, 0, 1], None, 'hero')
        self.objects.append(self.hero)

        # randomly set a goal position
        self.goal = gameOb(self.newPosition(0), 1, [1, 0, 0], 1, 'goal')
        self.objects.append(self.goal)

        # init wall
        self.initWall()

        # init timestep
        self.timestep = 0
        self.time = 0

        # state
        hero_ind = self.hero.x * 4 + self.hero.y
        state = tf.one_hot(hero_ind, self.state_dim, dtype=tf.float32)
        # self.state = tf.identity(state)
        self.bg = self.renderEnv()
        self.frame = self.renderAll()

        return state

    def reset_trial(self):
        # reset state after each trial
        self.objects = []

        # randomly set a hero position
        self.hero = gameOb(self.newPosition(0), 1, [0, 0, 1], None, 'hero')
        self.objects.append(self.hero)

        # randomly set a goal position
        self.goal = gameOb(self.newPosition(0), 1, [1, 0, 0], 1, 'goal')
        self.objects.append(self.goal)

        hero_ind = self.hero.x * 4 + self.hero.y
        state = tf.one_hot(hero_ind, self.state_dim, dtype=tf.float32)
        # self.state = tf.identity(state)
        self.bg = self.renderEnv()
        self.frame = self.renderAll()

        return state

    def moveChar(self, action):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise

        # blockPositions = [[-1,-1]]
        # for ob in self.objects:
        #     if ob.name == 'block': blockPositions.append([ob.x,ob.y])
        # blockPositions = np.array(blockPositions)
        heroX = self.hero.x
        heroY = self.hero.y
        hero_ori = self.hero.x * 4 + self.hero.y
        state = tf.one_hot(hero_ori, self.state_dim, dtype=tf.float32)

        # up
        if action == 0:
            heroX = (heroX - 1) % self.sizeX
        # down
        elif action == 1:
            heroX = (heroX + 1) % self.sizeX
        # left
        elif action == 2:
            heroY = (heroY - 1) % self.sizeY
        # right
        elif action == 3:
            heroY = (heroY + 1) % self.sizeY
        hero_now = heroX * 4 + heroY

        if self.wall[hero_ori, hero_now] == 1:  # no wall
            self.hero.x = heroX
            self.hero.y = heroY
            self.objects[0] = self.hero
            state = tf.one_hot(hero_now, self.state_dim, dtype=tf.float32)
        return state

    def newPosition(self, sparcity):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        for objectA in self.objects:
            if (objectA.x, objectA.y) in points: points.remove((objectA.x, objectA.y))
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        hero = self.objects[0]
        goal = self.objects[1]  # goal
        # goal = self.goal
        ended = False

        if hero.x == goal.x and goal.y == hero.y and hero != goal:
            self.objects.remove(goal)  # hit goal
            # self.objects.append(gameOb(self.newPosition(0),1,self.goal_color,1,'goal'))
            return goal.reward, True
        # if nothing strikes into me, no goal reached
        if ended == False:
            return 0.0, False

    def renderAll(self):
        a = self.bg.copy()
        for item in self.objects:
            itemX = item.x * 60 + 35
            itemY = item.y * 60 + 35
            a[itemX - 25:itemX + 25, itemY - 25:itemY + 25, :] = item.color
        return a

    def renderEnv(self):
        a = np.ones([250, 250, 3])
        # grid
        for i in range(self.sizeX + 1):
            pos = i * 60 + 5
            a[pos - 2:pos + 2, :, :] = [50, 50, 50]
            a[:, pos - 2:pos + 2, :] = [50, 50, 50]

        # wall
        wall = self.wall.copy()
        where_0 = np.where(wall == 0)
        where_1 = np.where(wall == 1)

        wall[where_0] = 1
        wall[where_1] = 0
        src, dst = np.nonzero(wall)

        for s, d in zip(src, dst):
            if s == d: continue
            m = np.min([s, d])
            n = np.max([s, d])
            if m + 3 == n:
                # same row and recurrent
                row_ind = m // 4
                start = 5 + row_ind * 60
                a[start:start + 60, 0:10, :] = [0, 0, 0]
                a[start:start + 60, 240:, :] = [0, 0, 0]
            elif m + 3 > n:
                row_ind = m // 4
                row_start = 5 + row_ind * 60
                colum_ind = m % 4 + 1
                colum_start = colum_ind * 60
                a[row_start:row_start + 60, colum_start:colum_start + 10, :] = [0, 0, 0]
            elif m + 12 == n:
                # same colum and recurrent
                colum_ind = m % 4
                colum_start = 5 + colum_ind * 60
                a[0:10, colum_start:colum_start + 60, :] = [0, 0, 0]
                a[240:, colum_start:colum_start + 60, :] = [0, 0, 0]
            else:
                colum_ind = m % 4
                colum_start = 5 + colum_ind * 60
                row_ind = m // 4 + 1
                row_start = row_ind * 60
                a[row_start:row_start + 10, colum_start:colum_start + 60, :] = [0, 0, 0]

        return a

    def step(self, action):
        state = self.moveChar(action)
        reward, trial_done = self.checkGoal()  # check whether reach goal
        done = False

        self.timestep += 1
        self.time += 400
        if self.time >= 20000:  # whether episode done
            done = True
        return state, reward, done, trial_done, self.timestep, self.time

    def replay(self):
        self.time += 120
        return self.time


import pygame, math, random, copy, cPickle, os
from Box2D import *

###########
# classes #
###########

class RandomGround(object):
    def __init__(self, world, pieces, groundDef = None):
        self.pieces = pieces
        smallPieceSize = (10, 1)
        self.smallPieceVertices = self.getPieceVertices(smallPieceSize)
        largePieceSize = (40, 1)
        self.largePieceVertices = self.getPieceVertices(largePieceSize)
        self.position = b2Vec2((0,10))
        if groundDef == None:
            startAngle = 0.1
            angleIncrement = 0.025
            angleRange = [startAngle + i* angleIncrement for i in
                range(self.pieces)]
            self.angles = [random.uniform(-x, x) for x in angleRange]
        else:
            self.angles = groundDef
        self.createBody(world)

    def getPieceVertices(self, pieceSize):
        pass
        return [(0,0),(pieceSize[0],0),(pieceSize[0],-pieceSize[1]),
            (0,-pieceSize[1])]

    def createBody(self, world):
        self.ground = world.CreateStaticBody(position = self.position)
        self.createStartWall()
        self.createPieces()
        self.createEndWall()

    def createStartWall(self):
        transform = b2Transform((0,0), b2Rot(math.pi/2))
        transformedVertices = [transform * b2Vec2(v) for v in
            self.largePieceVertices]
        startWall = b2FixtureDef(shape = b2PolygonShape(vertices =
            transformedVertices))
        self.ground.CreateFixture(startWall)
    def createPieces(self):
        self.transform = b2Transform((0,0), b2Rot(0))
        for i in range(self.pieces):
            transformedVertices = [self.transform * b2Vec2(v) for v in
                self.smallPieceVertices]
            pieceShape = b2PolygonShape(vertices = transformedVertices)
            self.transform = b2Transform(transformedVertices[1],
                b2Rot(self.angles[i]))
            pieceFixture = b2FixtureDef(shape = pieceShape)
            self.ground.CreateFixture(pieceFixture)
    def createEndWall(self):
        transform = b2Transform(self.transform.position, b2Rot(math.pi/2))
        transformedVertices = [transform * b2Vec2(v) for v in
            self.largePieceVertices]
        endWall = b2FixtureDef(shape = b2PolygonShape(vertices =
            transformedVertices))
        self.ground.CreateFixture(endWall)

    def getDef(self):
        return copy.copy(self.angles)

class CarChromosome(object):

    # example for vertices = 8, wheels = 2
    # chromosome has 23 variables, i.e. genes
    # gene | defines
    # -----------------------------------------
    # 0    | vertex 1 magnitude
    # ...  | ...
    # 7    | vertex 8 magnitude
    # 8    | vertex 1 angle
    # ...  | ...
    # 15   | vertex 8 angle
    # 16   | body density
    # 17   | wheel 1 radius
    # 18   | wheel 2 radius
    # 19   | wheel 1 density
    # 20   | wheel 2 density
    # 21   | wheel 1 pos
    # 22   | wheel 2 pos

    def __init__(self, vertices, wheels, genes = None):
        self.vertices, self.wheels = vertices, wheels
        self.vertexMagnitudeRange = (1, 5)
        self.vertexAngleRange = (-math.pi/self.vertices, math.pi/self.vertices)
        self.bodyDensityRange = (0.5, 2)
        self.wheelRadiusRange = (1, 3)
        self.wheelDensityRange = (0.5, 2)
        self.wheelPosRange = (0, self.vertices)
        self.wheelFrictionRange = (5, 20)
        if genes == None:
            self.genes = []
            for i in range(2*self.vertices + 3*self.wheels + 1):
                self.genes.append(self.getRandomGene(i))
        else:
            self.genes = genes

    def getGeneRange(self, i):
        if i < self.vertices:
            return self.vertexMagnitudeRange
        elif i < 2*self.vertices:
            return self.vertexAngleRange
        elif i < 2*self.vertices + 1:
            return self.bodyDensityRange
        elif i < 2*self.vertices + 1 + self.wheels:
            return self.wheelRadiusRange
        elif i < 2*self.vertices + 1 + 2*self.wheels:
            return self.wheelDensityRange
        elif i < 2*self.vertices + 1 + 3*self.wheels:
            return self.wheelPosRange

    def getRandomGene(self, i):
        if i >= 2*self.vertices + 1 + 2*self.wheels:
            return int(math.floor(random.uniform(*self.getGeneRange(i))))
        return random.uniform(*self.getGeneRange(i))

class Car(object):
    def __init__(self, pos, vertices, wheels, genes = None):
        self.spawnX = pos[0]
        if genes == None:
            self.chromosome = CarChromosome(vertices, wheels)
        else:
            self.chromosome = CarChromosome(vertices, wheels, genes)
        self.pos = pos
        self.vertices, self.wheels = vertices, wheels
        self.spawned = False
    
    def spawn(self, world):
        self.spawned = True
        self.initChassis(world, self.pos,
            self.chromosome.genes[:2*self.vertices+1])
        self.initWheels(world, self.chromosome.genes[2*self.vertices+1:])

    def deSpawn(self, world):
        self.spawned = False
        for axle in self.axleList:
            world.DestroyJoint(axle)
        world.DestroyBody(self.chassis)
        for wheel in self.wheelList:
            world.DestroyBody(wheel)

    def initChassis(self, world, pos, chassisGenes):
        restitution = 0.2
        friction = 10
        filterGroup = -8

        magnitudes = chassisGenes[:self.vertices]
        angles = chassisGenes[self.vertices:self.vertices*2]
        self.verticesList = [polarToRectangular(magnitudes[i], angles[i] +
            i*2*math.pi/self.vertices) for i in range(self.vertices)]
        self.chassis = world.CreateDynamicBody(position = pos)
        for i in range(self.vertices):
            fix = self.chassis.CreateFixture(b2FixtureDef(shape =
                b2PolygonShape(vertices = [(0,0), self.verticesList[i],
                self.verticesList[i-1]]),
                density = chassisGenes[self.vertices*2], friction = friction,
                restitution = restitution))
            fix.filterData.groupIndex = filterGroup

        self.totalMass = self.chassis.mass

    def initWheels(self, world, wheelsGenes):
        restitution = 0.2
        friction = 50
        filterGroup = -8

        self.wheelList = [None] * self.wheels
        self.axleList = [None] * self.wheels

        for i in range(self.wheels):
            self.wheelList[i] = world.CreateDynamicBody(position = (0,0))
            fix = self.wheelList[i].CreateCircleFixture(radius = wheelsGenes[i],
                density = wheelsGenes[self.wheels + i], friction = friction,
                restitution = restitution)
            fix.filterData.groupIndex = filterGroup
            self.totalMass += self.wheelList[i].mass
        self.createJoints(world, wheelsGenes)

    def createJoints(self, world, wheelsGenes):
        wheelSpeed = 100
        for i in range(self.wheels):
            wheelPos = (self.chassis.position +
                self.verticesList[wheelsGenes[self.wheels*2 + i]])
            wheelAngularSpeed = (wheelSpeed/
                self.wheelList[i].fixtures[0].shape.radius)
            wheelTorque = (self.totalMass*-world.gravity.y*
                self.wheelList[i].fixtures[0].shape.radius)
            self.wheelList[i].position = wheelPos
            self.axleList[i] = world.CreateRevoluteJoint(bodyA = self.chassis,
                bodyB = self.wheelList[i], anchor = wheelPos,
                maxMotorTorque = wheelTorque,
                motorSpeed = -wheelAngularSpeed,
                enableMotor = True)

    def getPos(self):
        if self.spawned:
            return self.chassis.worldCenter
        return None

    def getVel(self):
        if self.spawned:
            return self.chassis.linearVelocity
        return None

    def getDef(self):
        pass
        return copy.copy(self.chromosome.genes)

class Population(object):
    def __init__(self, world, individualClass, popSize, mutationRate,
        crossoverRate, keepElite, popDef = None, *args):
        self.generation = 0
        self.individualClass = individualClass
        self.popSize = popSize
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.keepElite = keepElite
        self.args = list(args)
        self.newIndividuals = []
        if popDef == None:
            for i in range(self.popSize):
                self.addNewIndividual()
        else:
            for i in range(self.popSize):
                genes = popDef[i]
                self.addNewIndividual(genes)
        self.individuals = self.newIndividuals
        self.fitnesses = [None] * self.popSize
        self.spawnAll(world)
        self.message = ""

    def addNewIndividual(self, genes = None):
        pass
        self.newIndividuals.append(self.individualClass(*(self.args + [genes])))

    def reportFitness(self, world, i, fitness):
        self.fitnesses[i] = fitness
        if self.allFitnessesGotten():
            prevDef = self.getDef()
            self.message = ("gen: " + str(self.generation) + 
                " avg: " + str(int(round(sum(self.fitnesses)/self.popSize))) +
                " max: " + str(int(round(max(self.fitnesses)))))
            self.createNextGeneration(world)
            return prevDef, self.message
        return None, None

    def allFitnessesGotten(self):
        for fitness in self.fitnesses:
            if fitness == None: return False
        return True

    def createNextGeneration(self, world):
        self.generation += 1
        self.newIndividuals = []
        tempFitnesses = copy.copy(self.fitnesses)
        eliteIndices = []
        for i in range(self.keepElite):
            maxIndex = tempFitnesses.index(max(tempFitnesses))
            eliteIndices.append(maxIndex)
            self.newIndividuals.append(self.individuals[maxIndex])
            tempFitnesses[maxIndex] = -1
        self.message += "\nelite: " + ", ".join([str(x) for x in eliteIndices])
        self.message += "\ncrossover pairs:"
        self.tournamentEvolve()
        # self.rouletteEvolve()
        self.individuals = self.newIndividuals
        self.fitnesses = [None] * self.popSize
        self.spawnAll(world)

    def tournamentEvolve(self):
        while len(self.newIndividuals) < self.popSize:
            genes1, genes2 = self.crossoverMutation(*self.tournamentSelect())
            self.addNewIndividual(genes1)
            self.addNewIndividual(genes2)
            if self.popSize - len(self.newIndividuals) == 1:
                self.newIndividuals.pop()
    
    def createTournament(self, size, exclude = None):
        indices = list(range(self.popSize))
        if exclude != None: indices.remove(exclude)
        tournament = []
        while len(tournament) < size:
            i = random.choice(indices)
            tournament.append(i)
            indices.remove(i)
        return tournament

    def getWinner(self, tournament):
        return max(tournament, key = lambda i: self.fitnesses[i])

    def tournamentSelect(self):
        size = 3
        i1 = self.getWinner(self.createTournament(size))
        i2 = self.getWinner(self.createTournament(size, exclude = i1))
        return i1, i2

    def rouletteEvolve(self):
        c = cProb(self)
        while len(self.newIndividuals) < self.popSize:
            genes1, genes2 = self.crossoverMutation(*self.rouletteSelect(c))
            self.addNewIndividual(genes1)
            self.addNewIndividual(genes2)
            if self.popSize < len(self.newIndividuals) == 1:
                self.newIndividuals.pop()

    def rouletteSelect(self, c):
        i1 = self.roulette(c)
        i2 = self.roulette(c, i1)
        return i1, i2

    def cProb(self):
        totalFitness = sum(self.fitnesses)
        p = []
        for i in range(self.popSize):
            p.append(self.fitnesses[i]/totalFitness)
        return [sum(p[:i+1]) for i in range(self.popSize)]

    def roulette(self, c, exclude = None):
        if exclude != None:
            dif = c[exclude] if exclude == 0 else c[exclude] - c[exclude-1]
            tempC = copy.copy(c)
            for i in range(exclude, self.popSize):
                tempC[i] -= dif
            r = random.random() * (1-dif)
        else:
            r = random.random()
            tempC = c
        for i in range(self.popSize):
            if r < tempC[i] and (i == 0 or r >= tempC[i-1]):
                return i

    def crossoverMutation(self, i1, i2):
        self.message += "\n" + str(i1) + "><" + str(i2)
        chrom1 = self.individuals[i1].chromosome
        chrom2 = self.individuals[i2].chromosome
        genes1, genes2 = self.crossover(chrom1, chrom2)
        return self.mutation(genes1, genes2, chrom1)
    def crossover(self, chrom1, chrom2):
        genes1 = []
        genes2 = []
        for i in range(len(chrom1.genes)):
            if random.random() < self.crossoverRate:
                genes1.append(chrom2.genes[i])
                genes2.append(chrom1.genes[i])
            else:
                genes1.append(chrom1.genes[i])
                genes2.append(chrom2.genes[i])
        return genes1, genes2
    def mutation(self, genes1, genes2, chrom1):
        for i in range(len(chrom1.genes)):
            if random.random() < self.mutationRate:
                genes1[i] = chrom1.getRandomGene(i)
            if random.random() < self.mutationRate:
                genes2[i] = chrom1.getRandomGene(i) # should be same as chrom2
        return genes1, genes2

    def spawnAll(self, data):
        for individual in self.individuals:
            individual.spawn(data)

    def getDef(self):
        pass
        return [individual.getDef() for individual in self.individuals]

class ValueSetter(object):
    def __init__(self, font, pos, label, var = 0):
        self.selected = False
        self.pos = pos
        self.label = label
        self.var = str(var)
        self.font = font
        self.text = self.font.render(self.label + ": " + self.var , True,
            (0, 0, 0))

    def draw(self, screen):
        pass
        screen.blit(self.text, self.pos)

    def input(self, ch):
        self.var += ch
        self.text = self.font.render(self.label + ": " + self.var , True,
            (0, 0, 0))

    def backspace(self):
        self.var = self.var[:-1]
        self.text = self.font.render(self.label + ": " + self.var , True,
            (0, 0, 0))

    def getBox(self):
        pass
        return self.text.get_rect().move(*self.pos)

    def getValue(self):
        if "." in self.var:
            return float(self.var)
        return int(self.var)

class Struct(object): pass

########
# init #
########
def newPop(data, popDef = None):
    return Population(data.world, Car, data.settings["population size"],
        data.settings["mutation rate"], data.settings["crossover rate"],
        data.settings["keep elite"], popDef, (data.settings["spawn x"],
        data.settings["spawn y"]), data.settings["vertices"],
        data.settings["wheels"])
def newGround(data, groundDef = None):
    pass
    return RandomGround(data.world, data.settings["ground pieces"], groundDef)

def init(settings = None, toggles = None, popDef = None, groundDef = None):
    data = Struct()
    pygame.init()
    quit = False
    data.editMode = False
    if settings == None:
        quit = initSettings(data)
        popDef, groundDef = None, None
    else: data.settings = settings
    if quit: return
    if data.editMode: editor(data)
    else:
        initPygame(data)
        initBox2D(data)
        initUI(data, toggles)
        initPlay(data, popDef, groundDef)
        play(data)

def initSettings(data):
    data.settings = {"vertices": 5, "wheels": 3, "population size": 10,
        "mutation rate": 0.1, "crossover rate": 0.7, "keep elite": 3,
        "gravity": 200}
    (data.settings["spawn x"], data.settings["spawn y"]) = (15, 25)
    data.settings["stale frames"] = 100
    data.settings["width"], data.settings["height"] = 50, 30
    data.settings["scale"] = 10.0
    data.settings["max fps"] = 100
    data.settings["stale min speed"] = 2.0
    data.settings["ground pieces"] = 50
    data.settings["zoom factor"] = 1.1
    data.settings["zoom bound"] = 20
    data.settings["cam speed"] = 2
    return editSettings(data)
def initPygame(data):
    data.clock = pygame.time.Clock()
    data.screen = pygame.display.set_mode(
        (int(data.settings["scale"]*data.settings["width"]),
        int(data.settings["scale"]*data.settings["height"])))
def initBox2D(data):
    data.world = b2World()
    data.world.gravity = b2Vec2(0,-data.settings["gravity"])
    data.timeStep = 1.0 / data.settings["max fps"]
    data.vel_iters, data.pos_iters = 20,10
def initPlay(data, popDef = None, groundDef = None):
    data.paused = False
    data.playing = True
    data.pop = newPop(data, popDef)
    data.ground = newGround(data, groundDef)
    data.genDefs = []
    data.posList = [0]*data.settings["population size"]
    data.velList = [0]*data.settings["population size"]
    data.bestPosList = [0]*data.settings["population size"]
    data.staleList = [0]*data.settings["population size"]
    data.followLeader = True
    data.followCar = None
    data.camera = [data.settings["width"]/2,data.settings["height"]/2]
    data.zoomLevel = 0
    data.controls = {pygame.K_MINUS: False, pygame.K_EQUALS: False,
        pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False,
        pygame.K_RIGHT: False}

def editSettings_dispatchEvents(editSettingsData):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            editSettings_mouse(editSettingsData, event, True)
        elif event.type == pygame.MOUSEBUTTONUP:
            editSettings_mouse(editSettingsData, event, False)
        elif event.type == pygame.KEYDOWN:
            editSettings_key(editSettingsData, event, True)
        elif event.type == pygame.KEYUP:
            editSettings_key(editSettingsData, event, False)
        elif event.type == pygame.QUIT:
            editSettingsData.playing = False
            editSettingsData.quit = True
            pygame.quit()
def editSettings_mouse(editSettingsData, event, down):
    if down:
        for setter in editSettingsData.setters:
            if setter.getBox().collidepoint(event.pos):
                editSettingsData.selection = setter
                return
        editSettingsData.selection = None
def editSettings_key(editSettingsData, event, down):
    if down:
        if editSettingsData.selection != None:
            if event.key in list(range(pygame.K_0, pygame.K_9 + 1)):
                editSettingsData.selection.input(event.unicode)
            if event.key == pygame.K_PERIOD:
                editSettingsData.selection.input(event.unicode)
            if event.key == pygame.K_BACKSPACE:
                editSettingsData.selection.backspace()
        if event.key == pygame.K_RETURN:
            editSettingsData.editMode = False
            editSettingsData.playing = False
        if event.key == pygame.K_e:
            editSettingsData.editMode = True
            editSettingsData.playing = False

def setData(data, editSettingsData):
    data.editMode = editSettingsData.editMode
    for i in range(len(editSettingsData.setterArgs)):
        setValue = editSettingsData.setters[i].getValue()
        assert(type(data.settings[editSettingsData.setterArgs[i]]) ==
            type(setValue))
        data.settings[editSettingsData.setterArgs[i]] = setValue
def initSetters(data, editSettingsData):
    font = pygame.font.Font("Courier New.ttf", 20)
    spacing = font.get_linesize()
    editSettingsData.setters = []
    editSettingsData.setterArgs = ["vertices", "wheels", "population size",
    "mutation rate", "crossover rate", "keep elite", "ground pieces", "gravity"]
    for i in range(len(editSettingsData.setterArgs)):
        editSettingsData.setters.append(ValueSetter(font, (0,i*spacing),
            editSettingsData.setterArgs[i],
            data.settings[editSettingsData.setterArgs[i]]))
    editSettingsData.selection = None
def drawSetters(editSettingsData):
    if editSettingsData.selection != None:
        pygame.draw.rect(editSettingsData.screen, (200,200,200),
            editSettingsData.selection.getBox())
    for setter in editSettingsData.setters:
        setter.draw(editSettingsData.screen)

def crashRepeatWrapper(f):
    def g(*args):
        try: f(*args)
        except:
            print "crashed! try again"
            g(*args)
    return g
@crashRepeatWrapper
def editSettings(data):
    editSettingsData = Struct()
    editSettingsData.quit = False

    editSettingsData.clock = pygame.time.Clock()
    editSettingsData.screen = pygame.display.set_mode((250, 190))

    editSettingsData.playing = True
    initSetters(data, editSettingsData)

    while editSettingsData.playing:
        # draw
        editSettingsData.screen.fill((255,255,255))
        drawSetters(editSettingsData)

        #update
        pygame.display.update()
        editSettingsData.time = editSettingsData.clock.tick(30)

        # events
        editSettings_dispatchEvents(editSettingsData)
    if not editSettingsData.quit: setData(data, editSettingsData)
    return editSettingsData.quit

########
# play #
########

# draw

def polarToRectangular(r, theta):
    pass
    return b2Vec2(math.cos(theta) * r, math.sin(theta) * r)
def rectangularToPolar(x, y):
    pass
    return distance((x,y), (0,0)), math.atan2(-y, x)
def distance(pos1, pos2):
    pass
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def calculateZoom(data):
    pass
    data.zoom = data.settings["zoom factor"] ** data.zoomLevel

def drawGrid(data, color = (240,240,240), step = 1, width = 1):
    width = int(round(width * data.zoom))
    xBounds = (data.camera[0] - data.settings["width"]/2/data.zoom,
               data.camera[0] + data.settings["width"]/2/data.zoom)
    yBounds = (data.camera[1] - data.settings["height"]/2/data.zoom,
               data.camera[1] + data.settings["height"]/2/data.zoom)
    for i in filter(lambda x: x % step == 0, range(int(xBounds[0]),
        int(xBounds[1]) + 1)):
        drawLine(data, color, (i,yBounds[0]), (i,yBounds[1]), width)
    for i in filter(lambda x: x % step == 0, range(int(yBounds[0]),
        int(yBounds[1]) + 1)):
        drawLine(data, color, (xBounds[0],i), (xBounds[1],i), width)

def drawAngle(data, body):
    color = (255,0,0)
    width = 1
    length = 1
    pos2 = body.worldCenter + polarToRectangular(length, body.angle)
    drawLine(data, color, body.worldCenter, pos2, width)
def drawBody(data, body):
    drawAngle(data, body)
    for fixture in body.fixtures:
        if type(fixture.shape) == b2PolygonShape:
            drawPolygonFixture(data, fixture, body)
        elif type(fixture.shape) == b2CircleShape:
            drawCircleFixture(data, fixture, body)
def drawJoint(data, joint):
    color = (0,0,0)
    width = 1
    drawLine(data, color, joint.anchorA, joint.anchorB, width)

def drawPolygonFixture(data, fixture, body):
    color = (0,0,0)
    closed = True
    width = 1
    vertices = [body.transform * v for v in fixture.shape.vertices]
    drawLines(data, color, closed, vertices, width)
def drawCircleFixture(data, fixture, body):
    color = (0,0,0)
    width = 1
    position = body.transform * fixture.shape.pos
    drawCircle(data, color, position, fixture.shape.radius, width)

def scale(data, vec): # scales pybox2d coordinates to pygame coordinates
    if type(vec) == float or type(vec) == int:
        return int(vec * data.settings["scale"] * data.zoom)
    elif type(vec) == tuple or type(vec) == b2Vec2:
        vec = tuple(vec)
        box2dx = (data.settings["width"]/2 + (vec[0] - data.camera[0]) *
            data.zoom) # center on cam
        pygamex = box2dx * data.settings["scale"] # scale to pygame window
        box2dy = (data.settings["height"]/2 - (vec[1] - data.camera[1]) *
            data.zoom) # flip y and center on cam
        pygamey = box2dy * data.settings["scale"] # scale to pygame window
        return (int(pygamex), int(pygamey))

# easier to use draw funcs
def drawCircle(data, color, pos, rad, width):
    pygame.draw.circle(data.screen, color, scale(data, pos), scale(data, rad),
        width)
    pass
def drawLines(data, color, closed, vertices, width):
    pygame.draw.lines(data.screen, color, closed, [scale(data, x) for x in
        vertices], width)
    pass
def drawLine(data, color, pos1, pos2, width):
    pygame.draw.line(data.screen, color, scale(data, pos1), scale(data, pos2),
        width)
    pass

# draw sim elements
def drawBackground(data):
    data.screen.fill((255,255,255))
    drawGrid(data, step = 1, width = 1)
    drawGrid(data, step = 10, width = 5)
def drawBodies(data):
    for body in data.world.bodies:
        drawBody(data, body)
def drawJoints(data):
    for joint in data.world.joints:
        drawJoint(data, joint)

# draw UI
def initUI(data, toggles = None):
    data.font = pygame.font.Font("Courier New.ttf", 12)
    data.fontHeight = data.font.get_linesize()
    if toggles == None:
        data.toggles = {"disp help": True, "disp settings": False,
            "disp sim data": True, "disp car data": True}
    else:
        data.toggles = toggles
    initHelpScreen(data)
    initSettingsScreen(data)
def initHelpScreen(data):
    data.helpScreen = pygame.Surface((150, 225))
    data.helpScreen.fill((255,255,255))
    pygame.draw.rect(data.helpScreen, (0,0,0), data.helpScreen.get_rect(), 1)
    lines = ["help", "`: toggle pause", "h: toggle help", "s: toggle settings",
        "d: toggle sim data", "c: toggle car data", "0-9: follow car",
        "l: follow leader", "- =: zoom", "arrow keys: pan",
        "k: kill stalish cars", "p: new population", "g: new ground",
        "return: restart", ", .: load/save", "/: load car", "esc: quit"]
    for i in range(len(lines)): drawTextLine(data, data.helpScreen, lines[i], i)
    data.helpScreenPos = (
        (data.screen.get_width() - data.helpScreen.get_width())/2,
        (data.screen.get_height() - data.helpScreen.get_height())/2)
def initSettingsScreen(data):
    data.settingsScreen = pygame.Surface((150, 270), pygame.SRCALPHA,
        32).convert_alpha()
    lines = [(setting + ": " + str(data.settings[setting])) for setting in
        sorted(data.settings)]
    for i in range(len(lines)):
        drawTextLine(data, data.settingsScreen, lines[i], i)
    data.settingsScreenPos =(
        data.screen.get_width() - data.settingsScreen.get_width(),
        data.screen.get_height() - data.settingsScreen.get_height())
def drawUI(data):
    if data.toggles["disp help"]: drawHelp(data)
    if data.toggles["disp settings"]: drawSettings(data)
    if data.toggles["disp sim data"]: drawSimData(data)
    if data.toggles["disp car data"]: drawCarData(data)
def drawHelp(data):
    pass
    data.screen.blit(data.helpScreen, data.helpScreenPos)
def drawSettings(data):
    pass
    data.screen.blit(data.settingsScreen, data.settingsScreenPos)
def drawSimData(data):
    drawGenData(data)
    drawCamData(data)
    drawZoomData(data)
    drawFPSData(data)
def drawGenData(data):
    genText = data.font.render("generation: " + str(data.pop.generation),
        True, (0, 0, 0))
    data.screen.blit(genText, (data.screen.get_width()/2 -
        genText.get_width()/2, 0))
def drawCamData(data):
    camStr = "camera: "
    if data.followLeader:
        camStr = camStr + "leader (car " + str(data.followCar) + ")" 
    else:
        if data.followCar == None:
            camStr = camStr + str(int(round(data.camera[0]))) + ", " + str(
                int(round(data.camera[1])))
        else:
            camStr = camStr + "car " + str(data.followCar)
    camText = data.font.render(camStr, True, (0, 0, 0))
    data.screen.blit(camText, (data.screen.get_width()/2 -
        camText.get_width()/2, data.fontHeight))
def drawZoomData(data):
    zoomText = data.font.render("zoom level: " +
        str(data.zoomLevel) , True, (0, 0, 0))
    data.screen.blit(zoomText,
        (data.screen.get_width()/2 - zoomText.get_width()/2,
        2 * data.fontHeight))
def drawFPSData(data):
    fpsText = data.font.render("fps: " +
        str(int(round(data.clock.get_fps()))) , True, (0, 0, 0))
    data.screen.blit(fpsText, (data.screen.get_width() - fpsText.get_width(),
        0))
def drawCarData(data):
    colWidth = 30
    carDataScreen = pygame.Surface((200, 200), pygame.SRCALPHA, 32).convert_alpha()
    drawTextLine(data, carDataScreen, "car", 0, col = 0, colWidth = colWidth)
    drawTextLine(data, carDataScreen, "fit", 0, col = 1, colWidth = colWidth)
    drawTextLine(data, carDataScreen, "stale", 0, col = 2, colWidth = colWidth)
    for i in range(data.settings["population size"]):
        drawTextLine(data, carDataScreen, str(i), i+1, col = 0, colWidth = colWidth)
        if data.posList[i] != None:
            drawTextLine(data, carDataScreen, str(int(round(data.posList[i][0]))), i+1, col = 1, colWidth = colWidth)
            drawTextLine(data, carDataScreen, str(data.staleList[i]), i+1, col = 2, colWidth = colWidth)
        else:
            drawTextLine(data, carDataScreen, str(int(round(data.bestPosList[i]))), i+1, col = 1, colWidth = colWidth)
            drawTextLine(data, carDataScreen, "dead", i+1, col = 2, colWidth = colWidth)
    data.screen.blit(carDataScreen, (0,0))
def drawTextLine(data, surf, msg, line, col = 0, colWidth = 10):
    pass
    surf.blit(data.font.render(msg, True, (0, 0, 0)), (col * colWidth, line *
        data.fontHeight))

# events
def dispatchEvents(data):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse(data, event, True)
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse(data, event, False)
        elif event.type == pygame.KEYDOWN:
            key(data, event, True)
        elif event.type == pygame.KEYUP:
            key(data, event, False)
        if event.type == pygame.QUIT:
            data.playing = False
def mouse(data, event, down):
    if down:
        mousePos = event.pos
        distList = []
        minDist = 10
        for i in range(data.settings["population size"]):
            if data.posList[i] != None:
                carPos = scale(data, data.posList[i])
                distList.append(distance(mousePos, carPos))
            else:
                distList.append(None)
        closestDist = min(filter(lambda x: x != None, distList))
        closestI = distList.index(closestDist)
        if closestDist < minDist * data.settings["scale"] * data.zoom:
            data.followLeader = False
            data.followCar = closestI
def key(data, event, down):
    if event.key in data.controls:
        data.controls[event.key] = down
    elif down:
        if event.key == pygame.K_BACKQUOTE: data.paused = not data.paused
        elif event.key == pygame.K_h: toggle(data, "disp help")
        elif event.key == pygame.K_s: toggle(data, "disp settings")
        elif event.key == pygame.K_d: toggle(data, "disp sim data")
        elif event.key == pygame.K_c: toggle(data, "disp car data")
        elif event.key in list(range(pygame.K_0, pygame.K_9 + 1)):
            followCar(data, int(event.unicode)) 
        elif event.key == pygame.K_l: data.followLeader = True
        elif event.key == pygame.K_k: killStalishCars(data)
        elif event.key == pygame.K_p: regenPop(data)
        elif event.key == pygame.K_g: regenGround(data)
        elif event.key == pygame.K_RETURN: regenSettings(data)
        elif event.key == pygame.K_COMMA: save(data)
        elif event.key == pygame.K_PERIOD: load(data)
        elif event.key == pygame.K_SLASH: loadCar(data)
        elif event.key == pygame.K_ESCAPE: data.playing = False
def toggle(data, s):
    pass
    data.toggles[s] = not data.toggles[s]
def followCar(data, i):
    if data.posList[i] != None:
        data.followLeader = False
        data.followCar = i
def killStalishCars(data):
    for i in range(len(data.staleList)):
        if data.staleList[i] != None:
            data.staleList[i] = data.settings["stale frames"] - 1
def deSpawnCars(data):
    pass
    for car in data.pop.individuals: car.deSpawn(data.world)
def regenGround(data):
    init(settings = data.settings, toggles = data.toggles,
        popDef = data.pop.getDef(), groundDef = None)
def regenPop(data):
    init(settings = data.settings, toggles = data.toggles,
        popDef = None, groundDef = data.ground.getDef())
def regenSettings(data):
    pass
    init(toggles = data.toggles)
def save(data):
    saveDir = "saves"
    dirPrefix = "save"
    index = 0
    savePrefix = "gen"
    saveExt = ".sav"
    if not os.path.exists(os.curdir + os.sep + saveDir):
        os.mkdir(saveDir)
    while os.path.exists(os.curdir + os.sep + saveDir + os.sep + dirPrefix +
        str(index)):
        index += 1
    os.chdir(saveDir)
    os.mkdir(dirPrefix + str(index))
    os.chdir(dirPrefix + str(index))
    groundDef = data.ground.getDef()
    for i in range(len(data.genDefs)):
        d = [data.settings, data.toggles, data.genDefs[i], groundDef]
        cPickle.dump(d, open(savePrefix + str(i) + saveExt, "wb"))
    os.chdir("..")
    os.chdir("..")
def load(data):
    if os.path.exists("load.sav"):
        d = cPickle.load(open("load.sav", "rb"))
        init(*d)
    else:
        print "rename desired save as \"load.sav\" and place in current dir"
def loadCar(data):
    if os.path.exists("genes.sav"):
        genes = cPickle.load(open("genes.sav", "rb"))
        data.settings["population size"] = 1
        init(settings = data.settings, toggles = data.toggles,
        popDef = [genes], groundDef = data.ground.getDef())
    else:
        print "rename desired save as \"genes.sav\" and place in current dir"

def getCarPhysicsData(data):
    data.posList = [car.getPos() for car in data.pop.individuals]
    data.velList = [car.getVel() for car in data.pop.individuals]
def manageCars(data):
    getCarPhysicsData(data)
    for i in range(data.settings["population size"]):
        if data.velList[i] != None:
            if (data.velList[i][0] > data.settings["stale min speed"] and
                data.posList[i][0] > data.bestPosList[i]):
                data.bestPosList[i] = data.posList[i][0]
                data.staleList[i] = 0
            else:
                data.staleList[i] += 1
                if data.staleList[i] > data.settings["stale frames"]:
                    killCar(data, i)
def killCar(data, i):
    if i == data.followCar and not data.followLeader:
        data.followCar = None
    data.staleList[i] = None
    data.pop.individuals[i].deSpawn(data.world)
    genDef, message = data.pop.reportFitness(data.world, i, data.bestPosList[i])
    if genDef != None:
        print message
        data.genDefs.append(genDef)
        data.staleList = [0]*data.settings["population size"]
        data.bestPosList = [0]*data.settings["population size"]
def manageCamera(data):
    controls(data)
    if data.followLeader:
        data.followCar = getFurthestCar(data)
    if data.followCar != None:
        data.camera = list(data.posList[data.followCar])
def controls(data):
    for key in data.controls:
        if data.controls[key]:
            if key == pygame.K_EQUALS:
                data.zoomLevel = min(data.zoomLevel + 1, data.settings["zoom bound"])
            elif key == pygame.K_MINUS:
                data.zoomLevel = max(data.zoomLevel - 1, -data.settings["zoom bound"])
            else:
                data.followLeader = False
                data.followCar = None
                if key == pygame.K_UP:
                    data.camera[1] += data.settings["cam speed"] / data.zoom
                elif key == pygame.K_DOWN:
                    data.camera[1] -= data.settings["cam speed"] / data.zoom
                elif key == pygame.K_LEFT:
                    data.camera[0] -= data.settings["cam speed"] / data.zoom
                elif key == pygame.K_RIGHT:
                    data.camera[0] += data.settings["cam speed"] / data.zoom
def getFurthestCar(data):
    furthestPos = max(filter(lambda x: x != None, data.posList),
        key = lambda x: x[0])
    return data.posList.index(furthestPos)
def updatePygame(data):
    pygame.display.update()
    data.time = data.clock.tick(data.settings["max fps"])
def updatePhysics(data):
    pass
    data.world.Step(data.timeStep, data.vel_iters, data.pos_iters)

def play(data):
    data.playing = True
    while data.playing:
        # do stuff
        if not data.paused: manageCars(data)
        manageCamera(data)

        # draw
        calculateZoom(data)
        drawBackground(data)
        drawBodies(data)
        drawJoints(data)
        drawUI(data)

        # update
        updatePygame(data)
        if not data.paused: updatePhysics(data)

        # events
        dispatchEvents(data)
    pygame.quit()

# editor
@crashRepeatWrapper
def editor(data):
    editorData = Struct()
    editorClock = pygame.time.Clock()
    editorData.screen = pygame.display.set_mode((400, 600))

    editorData.playing = True
    editorData.chrom = CarChromosome(data.settings["vertices"], data.settings["wheels"])
    editorData.verticesList = [None] * data.settings["vertices"]
    editorData.genes = [None] * (2*data.settings["vertices"] +
        3*data.settings["wheels"] + 1)

    editor_initSetters(editorData)

    while editorData.playing:
        # draw
        editorData.screen.fill((255,255,255))

        drawBodyDesigner(editorData)
        drawVertices(editorData)
        editor_drawSetters(editorData)

        #update
        pygame.display.update()
        editorTime = editorClock.tick(30)

        # events
        editor_dispatchEvents(editorData)
    pygame.quit()
def editor_dispatchEvents(editorData):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            editor_mouse(editorData, event, True)
        elif event.type == pygame.MOUSEBUTTONUP:
            editor_mouse(editorData, event, False)
        elif event.type == pygame.KEYDOWN:
            editor_key(editorData, event, True)
        elif event.type == pygame.KEYUP:
            editor_key(editorData, event, False)
        if event.type == pygame.QUIT:
            editorData.playing = False
def editor_mouse(editorData, event, down):
    if down:
        for setter in editorData.setters:
            if setter.getBox().collidepoint(event.pos):
                editorData.selection = setter
                return
        setPoint(editorData, event)
def setPoint(editorData, event):
    mouseDistance, mouseAngle = rectangularToPolar(*b2Vec2(event.pos) -
        editorData.center)
    if mouseAngle < -math.pi/editorData.chrom.vertices:
        mouseAngle %= 2*math.pi
    if (mouseDistance >= editorData.minRadius and
        mouseDistance <= editorData.maxRadius):
        vertices = editorData.chrom.vertices
        angleIncrement = 2*math.pi/vertices
        angleOffset = math.pi/vertices
        i = (int(math.floor((mouseAngle + angleOffset)/(angleIncrement)))%
            editorData.chrom.vertices)
        editorData.verticesList[i] = event.pos
        maxMag = editorData.chrom.getGeneRange(0)[1]
        editorData.genes[i] = mouseDistance/editorData.maxRadius*maxMag
        editorData.genes[i + editorData.chrom.vertices] = (mouseAngle -
            i*angleIncrement)
def editor_key(editorData, event, down):
    if down:
        if editorData.selection != None:
            if event.key in list(range(pygame.K_0, pygame.K_9 + 1)):
                editorData.selection.input(event.unicode)
            if event.key == pygame.K_PERIOD:
                editorData.selection.input(event.unicode)
            if event.key == pygame.K_BACKSPACE:
                editorData.selection.backspace()
        if event.key == pygame.K_RETURN:
            createGenes(editorData)
            editorData.playing = False
def editor_initSetters(editorData):
    font = pygame.font.Font("Courier New.ttf", 20)
    spacing = font.get_linesize()
    editorData.selection = None
    editorData.bodyDensitySetter = ValueSetter(font, (0,0), "body density", 0)
    editorData.wheelSetters = []
    for i in range(editorData.chrom.wheels):
        editorData.wheelSetters.append(ValueSetter(font, (0,(3*i+1)*spacing),
            "wheel " + str(i) + " radius", 0))
        editorData.wheelSetters.append(ValueSetter(font, (0,(3*i+2)*spacing),
            "wheel " + str(i) + " density", 0))
        editorData.wheelSetters.append(ValueSetter(font, (0,(3*i+3)*spacing),
            "wheel " + str(i) + " position", 0))
    editorData.setters = [editorData.bodyDensitySetter] +editorData.wheelSetters
def editor_drawSetters(editorData):
    if editorData.selection != None:
        pygame.draw.rect(editorData.screen, (200,200,200),
            editorData.selection.getBox())
    for setter in editorData.setters:
        setter.draw(editorData.screen)
def drawBodyDesigner(editorData):
    color, width = (240, 240, 240), 1
    magRange = editorData.chrom.getGeneRange(0)
    editorData.center = (editorData.screen.get_width()/2, 400)
    editorData.maxRadius = 150
    editorData.minRadius = editorData.maxRadius/magRange[1]*magRange[0]
    pygame.draw.circle(editorData.screen, color, editorData.center,
        editorData.maxRadius, width)
    pygame.draw.circle(editorData.screen, color, editorData.center,
        editorData.minRadius, width)
    vertices = editorData.chrom.vertices
    for i in range(vertices):
        angleIncrement = 2*math.pi/vertices
        angleOffset = math.pi/vertices
        p1 = polarToRectangular(editorData.minRadius, i*angleIncrement +
            angleOffset) + editorData.center
        p2 = polarToRectangular(editorData.maxRadius, i*angleIncrement +
            angleOffset) + editorData.center
        pygame.draw.line(editorData.screen, color, p1, p2, width)
def drawVertices(editorData):
    color, radius, width = (0,0,0), 2, 1
    for i in range(len(editorData.verticesList)):
        if editorData.verticesList[i] != None:
            pygame.draw.circle(editorData.screen, color,
                editorData.verticesList[i], radius, width)
            pygame.draw.line(editorData.screen, color,
                editorData.verticesList[i], editorData.center, width)
            nexti = (i+1)%editorData.chrom.vertices
            if editorData.verticesList[nexti] != None:
                pygame.draw.line(editorData.screen, color,
                    editorData.verticesList[i], editorData.verticesList[nexti],
                    width)
def createGenes(editorData):
    editorData.genes[2*editorData.chrom.vertices] = (
        float(editorData.bodyDensitySetter.getValue()))
    for i in range(editorData.chrom.wheels):
        editorData.genes[1 + 2*editorData.chrom.vertices +
            i] = float(editorData.wheelSetters[3*i].getValue())
        editorData.genes[1 + 2*editorData.chrom.vertices +
            editorData.chrom.wheels + i] = float(editorData.wheelSetters[3*i +
            1].getValue())
        editorData.genes[1 + 2*editorData.chrom.vertices +
            2*editorData.chrom.wheels + i] = int(editorData.wheelSetters[3*i +
            2].getValue())
    for i in range(len(editorData.genes)):
        assert(editorData.genes[i] != None)
        geneRange = editorData.chrom.getGeneRange(i)
        assert(editorData.genes[i] >= geneRange[0])
        assert(editorData.genes[i] <= geneRange[1])
    editor_save(editorData.genes)
def editor_save(genes):
    saveDir = "editorsaves"
    savePrefix = "genes"
    index = 0
    saveExt = ".sav"
    if not os.path.exists(os.curdir + os.sep + saveDir):
        os.mkdir(saveDir)
    while os.path.exists(os.curdir + os.sep + saveDir + os.sep + savePrefix +
        str(index) + saveExt):
        index += 1
    os.chdir(saveDir)
    cPickle.dump(genes, open(savePrefix + str(index) + saveExt, "wb"))
    os.chdir("..")

# go go power rangers!!!
init()
import pygame
import time
import random
import os
import neat

RED = (250, 70, 80)
LIGHT_RED = (200, 40, 50)
WHITE = (255,255,255)

global score
score = 0

BALL_IMG = pygame.image.load(os.path.join("images", "ball.png"))
BACKGROUND = pygame.image.load(os.path.join("images", "background.png"))
PLATFORM_IMG = pygame.image.load(os.path.join("images", "platform.png"))
WALL_BLOCK = pygame.image.load(os.path.join("images", "wall.png"))

WIDTH = 780
HEIGHT = 480

PLATFORM_WIDTH = 70
PLATFORM_HEIGHT = 20

GROUND = HEIGHT - PLATFORM_HEIGHT

BORDER_WIDTH = 60
BALL_RADIUS = 15

PLATFORM_POS = [y for y in range(BORDER_WIDTH // 2, WIDTH - PLATFORM_WIDTH - BORDER_WIDTH // 2)]


def drawWall(win):
	# win.blit(BACKGROUND, (0, 0)) #win.fill(RED)
	# pygame.draw.rect(win, LIGHT_RED, (0, 0, WIDTH, HEIGHT + BORDER_WIDTH), BORDER_WIDTH)
	for i in range(HEIGHT// (BORDER_WIDTH//2)):
		win.blit(WALL_BLOCK, (0, i * (BORDER_WIDTH//2)))
		win.blit(WALL_BLOCK, (WIDTH - BORDER_WIDTH//2, i * (BORDER_WIDTH//2)))

	for i in range(1, WIDTH//(BORDER_WIDTH//2) - 1):
		win.blit(WALL_BLOCK, (i * (BORDER_WIDTH//2), 0))
	pygame.display.update()



class Ball:
	def __init__(self, platform):
		self.y = platform.y + PLATFORM_WIDTH // 2
		self.x = GROUND - BALL_RADIUS
		self.prX = self.x
		self.prY = self.y
		self.dx = 1
		self.score = 0
		self.dy = random.choice([-1, 1])
		

	def draw(self, win):
		# pygame.draw.circle(win, RED, (self.prY, self.prX), BALL_RADIUS)
		# pygame.draw.circle(win, WHITE, (self.y, self.x), BALL_RADIUS)
		win.blit(BALL_IMG, (self.y, self.x))
		drawWall(win)
		# pygame.draw.rect(win, LIGHT_RED, (0, 0, WIDTH, HEIGHT + BORDER_WIDTH), BORDER_WIDTH)
		pygame.display.update()

	def drawEnd(self, win):
		pygame.draw.circle(win, RED, (self.y, self.x), BALL_RADIUS)
		pygame.draw.rect(win, LIGHT_RED, (0, 0, WIDTH, HEIGHT + BORDER_WIDTH), BORDER_WIDTH)
		pygame.display.update()

	def move(self, platform):
		self.prX = self.x
		self.prY = self.y

		if self.y + BALL_RADIUS >= WIDTH - (BORDER_WIDTH // 2):
			self.dy = -1

		elif self.y - BALL_RADIUS <= BORDER_WIDTH // 2:
			self.dy = 1

		elif self.x - BALL_RADIUS <= BORDER_WIDTH // 2:
			self.dx = 1

		elif self.x + BALL_RADIUS == GROUND:
			if self.y <= platform.y + PLATFORM_WIDTH and self.y >= platform.y:
				self.dx = -1

		
		self.x += 2*self.dx
		self.y += 2*self.dy

class PLatform:
	def __init__(self):
		self.x = GROUND
		self.y = random.choice(PLATFORM_POS)
		self.prX = self.x
		self.prY = self.y
		self.vel = 5

	def draw(self, win):
		# pygame.draw.rect(win, RED, (self.prY, self.prX, PLATFORM_WIDTH, PLATFORM_HEIGHT), 0)
		# pygame.draw.rect(win, WHITE, (self.y, self.x, PLATFORM_WIDTH, PLATFORM_HEIGHT), 0)
		win.blit(PLATFORM_IMG, (self.y, self.x))
		drawWall(win)
		# pygame.draw.rect(win, LIGHT_RED, (0, 0, WIDTH, HEIGHT + BORDER_WIDTH), BORDER_WIDTH)
		pygame.display.update()

	def drawEnd(self, win):
		pygame.draw.rect(win, RED, (self.y, self.x, PLATFORM_WIDTH, PLATFORM_HEIGHT), 0)
		pygame.draw.rect(win, LIGHT_RED, (0, 0, WIDTH, HEIGHT + BORDER_WIDTH), BORDER_WIDTH)
		pygame.display.update()

	def move(self, command):
		self.prX = self.x
		self.prY = self.y

		if command == -1:
			if self.y <= BORDER_WIDTH // 2:
				command = 0

		elif command == 1:
			if self.y + PLATFORM_WIDTH >= WIDTH - (BORDER_WIDTH // 2):
				command = 0

		self.y = self.y + command*self.vel


def eval_genomes(genomes, config):
	win = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	nets = []
	ge = []
	balls = []
	platforms = []

	for _,g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		g.fitness = 0
		ge.append(g)
		platforms.append(PLatform())

	for platform in platforms:
		balls.append(Ball(platform))

	win.blit(BACKGROUND, (0, 0))
	drawWall(win)

	run = True
	while run:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()

		for x, ball in enumerate(balls):

			ge[x].fitness += 0.1
			ball.move(platforms[x])

			output = nets[x].activate((ball.dx, ball.dy, abs(platforms[x].y - ball.y), abs(platforms[x].x - ball.x)))
			finalOP = output.index(max(output)) - 1

			platforms[x].move(finalOP)
			# platforms[x].draw(win)
			# ball.draw(win)

			

			if ball.x > platforms[x].x:
				ball.drawEnd(win)
				platforms[x].drawEnd(win)
				ge[x].fitness -= 5
				nets.pop(x)
				ge.pop(x)
				balls.pop(x)
				platforms.pop(x)

			elif ball.x == platforms[x].x - BALL_RADIUS:
				if ball.y >= platforms[x].y and ball.y <= platforms[x].y + PLATFORM_WIDTH:
					ge[x].fitness += 10
					ball.score += 1
				
				# if ge[x].fitness >= 10000:
				# 	print("SCORE -> {}".format(balls[x].score))
				# 	run = False
				# 	break

		win.blit(BACKGROUND, (0,0))
		drawWall(win)

		for ball in balls:
			ball.draw(win)

		for platform in platforms:
			platform.draw(win)
		pygame.display.update()
			

		if len(balls) == 0:
			run = False
			break



def run(config_path):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
						neat.DefaultSpeciesSet, neat.DefaultStagnation,
						config_path)

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(eval_genomes)

	print("Best fitness -> {}".format(winner))


if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-FeedForward.txt")
	run(config_path)

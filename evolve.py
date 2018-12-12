#!/usr/bin/env python
import random
import math
import operator
import argparse
import math
import numpy as np
from skimage import feature
from skimage.color import rgb2grey
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from gene import genetic, gene
#import util

TARGET_IMG_NAME = 'source.jpg'  # target image file name
POP_SIZE = 20                 # population size
MUT_RATE = 0.05               # mutation rate
GENERATIONS = 3000      	 # number of generations
CHILDREN_PER_GEN = 5          # children generated in each generations
SQUARE_SIZE = 50		  	# pixel size of each mosaic 
MUTATION_RATE = 0.1

TARGET = Image.open(TARGET_IMG_NAME).convert('RGB')
WIDTH, HEIGHT = TARGET.size

class Grid():

	def __init__(self, source_path, square_size = 15, parent_grid_1=None, parent_grid_2=None, orig_grid_map = None):
		self.source_path = source_path 
		
		self.source_image = Image.open(source_path).convert('RGB')

		self.pixels = square_size
		self.width, self.height = self.source_image.size

		# Get grid size 
		self.cols = self.width/self.pixels
		self.rows = self.height/self.pixels 

		# Crop the image if our pixels don't divide equally 
		self.source_image = self.source_image.crop((0, 0, self.cols*self.pixels, self.rows*self.pixels))

		# Find edges
		self.edg_img = self.source_image.filter(ImageFilter.UnsharpMask(2, percent=300))
		self.image_array = np.array(self.edg_img)
		self.img_edges = feature.canny(rgb2grey(self.image_array), sigma=2)

		# Create grid map and set values
		if parent_grid_1 is not None and parent_grid_2 is not None:
			self.grid_map = np.empty(shape=(self.rows, self.cols), dtype=object)
			op = genetic()
			for index, x in np.ndenumerate(self.grid_map):
				row = index[0]
				col = index[1]
				parent_1 = parent_grid_1[row, col]
				parent_2 = parent_grid_2[row, col]

				new_gene = op.crossover(parent_1, parent_2)
				new_gene = op.mutate(new_gene, rate = MUTATION_RATE) 
				new_gene.fitness = self.get_fitness_of_square(row,col,new_gene,self.source_image)
				population = orig_grid_map[row,col]
				population.append(new_gene)
				population.sort(key=lambda x: x.fitness) #sort population by fitness
				population = population[:POP_SIZE] #keep population small so computing is manageable

				self.grid_map[row,col] = population

		else: # initialize with random colors
			self.grid_map = np.empty(shape=(self.rows, self.cols), dtype=object)

			for index, x in np.ndenumerate(self.grid_map):
				row = index[0]
				col = index[1]

				# initialize gene
				g = gene()
				g.row = row
				g.col = col
				g.fitness = self.get_fitness_of_square(row,col,g,self.source_image)

				population = [] 
				population.append(g)
				self.grid_map[row,col] = population


		# set actual image
		self.im = self.get_current_img()
			
		# calculate total fitness
		self.fitness = self.get_total_fitness(self.source_image)

		# display image
		# self.im.show()

	def get_current_img(self):
		"""
		return im , image that can be saved to jpg 
		"""
		im = Image.new('RGB', ( self.source_image.size[0], self.source_image.size[1]))
		draw = ImageDraw.Draw(im, 'RGBA')
		
		for index, p in np.ndenumerate(self.grid_map):
			g = p[0]

			# Base background 
			x0 = g.col*SQUARE_SIZE
			y0 = g.row*SQUARE_SIZE
			x1 = x0 + SQUARE_SIZE - 1
			y1 = y0 + SQUARE_SIZE - 1
			xy = [(x0, y0), (x1, y1)]
			rgb = (g.color['r'], g.color['g'],g.color['b'], g.color['a'])
			draw.rectangle(xy,fill=(rgb))

			# First shape
			margin = SQUARE_SIZE / 8
			x0 = g.col*SQUARE_SIZE
			y0 = g.row*SQUARE_SIZE
			x1 = x0 + SQUARE_SIZE - 1
			y1 = y0 + SQUARE_SIZE - 1
			xy = [(x0 + margin, y0 + margin), (x1 - margin, y1 - margin)]
			rgb = (g.color2['r'], g.color2['g'],g.color2['b'], g.color2['a'])
			
			if g.shape == 'circle':
				draw.ellipse(xy,fill=(rgb))
			elif g.shape == 'square':
				draw.rectangle(xy,fill=(rgb))

			# Second Shape 
			margin = SQUARE_SIZE / 8
			x0 = g.col*SQUARE_SIZE
			y0 = g.row*SQUARE_SIZE
			x1 = x0 + SQUARE_SIZE - 1
			y1 = y0 + SQUARE_SIZE - 1
			xy = [(x0 + 3*margin, y0 + 3*margin), (x1 - 3*margin, y1 - 3*margin)]
			rgb = (g.color3['r'], g.color3['g'],g.color3['b'], g.color3['a'])

			if g.shape == 'circle':
				draw.ellipse(xy,fill=(rgb))
			elif g.shape == 'square':
				draw.rectangle(xy,fill=(rgb))

		del draw
		return im

	def get_fitness_of_square(self, row,col, g, source_image):
		"""
		Get fitness of the individual square.
		This is calculated by RMS distance of histograms.
		"""
		xy = (col*self.pixels, row*self.pixels, col*self.pixels+SQUARE_SIZE-1, row*self.pixels + SQUARE_SIZE -1) 
		source_block = self.source_image.crop(xy)

		im = Image.new('RGB', ( SQUARE_SIZE, SQUARE_SIZE))
		draw = ImageDraw.Draw(im, 'RGBA')
		rgb = (	(g.color['r'] + g.color2['r'] + g.color3['r'] )/ 3, 
				(g.color['g'] + g.color2['g'] + g.color3['g'] )/ 3,
				(g.color['b'] + g.color2['b'] + g.color3['b'] )/ 3,
				g.color['a']
			  )

		draw.rectangle((0,0,SQUARE_SIZE-1, SQUARE_SIZE-1),fill=(rgb))

		h = ImageChops.difference(source_block, im).histogram()
		fitness = math.sqrt(reduce(operator.add,
							map(lambda h, i: h*(i**2),
							h, range(256)*3)) /float(SQUARE_SIZE * SQUARE_SIZE))
	
		return fitness

	def get_total_fitness(self, source_image):
		"""
		Get total fitness of the grid.
		This was calculated by RMS distance of histograms.
		"""
		h = ImageChops.difference(source_image, self.im).histogram()
		return math.sqrt(reduce(operator.add,
							map(lambda h, i: h*(i**2),
							h, range(256)*3)) /  (float(self.source_image.size[0]) * self.source_image.size[1]))

	def save_current_img(self, f_name):
		"""
		Save image to a file.
		"""
		self.im.save(f_name, 'JPEG')

	def flatten_grid(self):
		"""
		flatten row x col grid into an array 
		"""
		grid_array = []
		for index, x in np.ndenumerate(self.grid):
			grid_array.append(x)

		return grid_array

def intialize(source_path):
	"""
	Randomly initializes grid of squares.
	"""
	# Initialize random generation of squares
	grid_array = []
	grid = Grid(source_path = source_path, square_size = SQUARE_SIZE)
	grid_array.append(grid)

	return grid_array


def evolve(grid_array, source_path):
	"""
	Take initial graph and evolve colors to match source image
	"""
	grid = grid_array[0] # get initial grid

	for i in xrange(GENERATIONS):
		parent_grid_1 = np.empty(shape=(grid.rows, grid.cols), dtype=object)
		parent_grid_2 = np.empty(shape=(grid.rows, grid.cols), dtype=object)

		for index, population in np.ndenumerate(grid.grid_map):
			
			# generate weighted choices according to fitness
			parent_choices = []

			w = 100
			for gene in population:
				parent_choices.append((gene, w))
				if w > 0:
					w = w - 10
			
			pop_choices = [val for val, cnt in parent_choices for j in range(cnt)]

			# generate matrix of parents for crossover
			row = index[0]
			col = index[1]
			parent_1 = random.choice(pop_choices)
			parent_2 = random.choice(pop_choices)
			parent_grid_1[row, col] = parent_1
			parent_grid_2[row, col] = parent_2

		# perform crossover by calling Grid()
		child_grid = Grid(source_path = source_path, square_size = SQUARE_SIZE, 
					 parent_grid_1 = parent_grid_1, parent_grid_2 = parent_grid_2,
					 orig_grid_map = grid.grid_map)

		grid_array.append(child_grid)
		grid_array.sort(key=lambda x: x.fitness)

		# print log info 
		if i % 10000 == 0 or i in [10,20,30,40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 5000]:
			grid_array[0].save_current_img(str(i)+'_b.png')  # save intermediate imgs
		if i % 10 == 0:
			# print current best fitness and avg fitness
			avg = sum(map(lambda x: x.fitness, grid_array)) / len(grid_array)
			print "Finish " + str(i) + " Current fitness: {} Average fitness: {}".format(grid_array[0].fitness, avg)


def main(): 
	# Need arguments 

	parser = argparse.ArgumentParser(description='Chuck Close generator')

	parser.add_argument('photos', metavar='N', type=str, nargs='+',
						help='Photo path')

	# # Not activated 
	# parser.add_argument("-c", "--color", default=False,
	# 					action='store', choices=["0", "1", "2"],
	# 					help="Specify color values")
	# # Not activated 
	# parser.add_argument("-d", "--diamond", default=False, action='store_true',
	# 					help="Use diamond grid instead of squares")

	args = parser.parse_args()

	if args.photos:
		source_path = args.photos[0]
	else:
		source_path = 'source.jpg'

	# Execute genetic algorithm
	grid_array = intialize(source_path)
	grid_array = evolve(grid_array, source_path)


if __name__ == '__main__':
	main() 

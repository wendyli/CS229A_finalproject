#!/usr/bin/env python

import random


class gene(object):
	"""
	The gene class, which represent a triangle.

	Attributes:
		color: The RGBA value of the triangle.
	"""
	def __init__(self):
		"""
		Inits a gene with mutation operation.
		"""
		self.mutate(row = 0, col = 0)

	def mutate(self, row = 0, col = 0):
		"""
		Mutation operation. Change the values of a gene.
		"""
		self.row = row
		self.col = col

		# background color
		self.color = {'r': random.randint(0, 50),
					  'g': random.randint(0, 255),
					  'b': random.randint(0, 255),
					  'a': 128}
		# second shape color
		self.color2 = {'r': random.randint(0, 255),
					   'g': random.randint(0, 50),
					   'b': random.randint(0, 255),
					   'a': 128}
		# third shape color
		self.color3 = {'r': random.randint(0, 255),
					   'g': random.randint(0, 255),
					   'b': random.randint(0, 50),
					   'a': 128}	
		self.shape = random.choice(['circle', 'square'])		  
		self.fitness = 0
		# Later draw the shapes 


class genetic(object):
	"""
	The genetic operation utils class.
	"""
	def crossover(self, parent_1, parent_2):
		"""
		Randomly choose a r, g, and b value from parent1 and parent2 
		"""
		new_gene = gene()
		new_gene.row = parent_1.row
		new_gene.col = parent_1.col
		new_gene.color = {'r': random.choice((parent_1.color['r'],parent_2.color['r'])),
						  'g': random.choice((parent_1.color['g'],parent_2.color['g'])),
						  'b': random.choice((parent_1.color['b'],parent_2.color['b'])),
						  'a': 128}
		new_gene.color2 = {'r': random.choice((parent_1.color2['r'],parent_2.color2['r'])),
						   'g': random.choice((parent_1.color2['g'],parent_2.color2['g'])),
						   'b': random.choice((parent_1.color2['b'],parent_2.color2['b'])),
						   'a': 128}
		
		new_gene.color3 = {'r': random.choice((parent_1.color3['r'],parent_2.color3['r'])),
						   'g': random.choice((parent_1.color3['g'],parent_2.color3['g'])),
						   'b': random.choice((parent_1.color3['b'],parent_2.color3['b'])),
						   'a': 128}

		return new_gene

	def mutate(self, gene, rate):
		"""
		Mutation Selector:
		Select some genes and let them mutate.
		"""
		if random.uniform(0, 1) < rate:
			gene.mutate(gene.row, gene.col)
		return gene

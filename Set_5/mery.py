# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:21:53 2024

@author: marysia_pop
"""

#%%
import numpy as np 
import math 
import csv 
import matplotlib.pyplot as plt
import matplotlib
#%%

def random_create_world(world_size):
    #world_size = 
    options = [0,1]
    world = np.zeros((world_size,world_size))
    for i in range (0,world_size):
        for j in range (0,world_size):
            world[i][j] = np.random.choice(options)
    return np.array(world).astype(int)


def calculate_N(world):
    N_calculated = np.zeros((len(world),len(world)))
    k=0
    for i in range (-1,2,1):
        for j in range (-1,2,1):
            N_calculated +=(2**k)*np.roll(np.roll(world,i,axis=0),j,axis=1)
            #print(i,j)
            k+=1
    return np.array(N_calculated).astype(int)

def generate_random_genome():
    r_genome = []
    options = [0,1]
    for i in range (0,512):
        r_genome.append(np.random.choice(options))
        #r_genome.append(1)
    #r_genome[256] = 0
    return np.array(r_genome).astype(int)

def fitness_func(world):
    #Substract -3 for same y+1 or x+1
    world_size=len(world)
    fitness_points = np.zeros((world_size,world_size))
    fitness_points_mask = np.zeros((world_size,world_size)).astype(bool)
    fitness_points_mask_1 = ~(np.logical_xor(world,np.roll(np.roll(world,0,axis=0),-1,axis=1)))
    fitness_points = np.ones((world_size,world_size))*(-3)*fitness_points_mask_1 + fitness_points
    fitness_points_mask_2 = ~np.logical_xor(world,np.roll(np.roll(world,-1,axis=0),0,axis=1))
    fitness_points = np.ones((world_size,world_size))*(-3)*fitness_points_mask_2 + fitness_points
    fitness_points_mask_3 = fitness_points_mask_1+ fitness_points_mask_2
    #Now otherwise
    fitness_points_mask_4 = ~np.logical_xor(world,np.roll(np.roll(world,-1,axis=0),-1,axis=1))
    fitness_points = np.ones((world_size,world_size))*(8)*(fitness_points_mask_4)*(~fitness_points_mask_3) +np.ones((world_size,world_size))*(-5)*(~fitness_points_mask_4)*(~fitness_points_mask_3)+ fitness_points
    fitness_points_mask_5 = ~np.logical_xor(world,np.roll(np.roll(world,-1,axis=0),1,axis=1))
    fitness_points = np.ones((world_size,world_size))*(8)*(fitness_points_mask_5)*(~fitness_points_mask_3) +np.ones((world_size,world_size))*(-5)*(~fitness_points_mask_5)*(~fitness_points_mask_3)+ fitness_points
    #print(np.sum(fitness_points))
    return np.sum(fitness_points)

def calculate_clone_fractions(results_of_all_genes):
    suma = np.sum(results_of_all_genes)
    return results_of_all_genes/suma

def uniform_crossover(gen1,gen2):
    options = [0,1]
    baby_gene = []
    for i in range (0,len(gen1)):
        which_one = np.random.choice(options)
        if (which_one == 0):
            baby_gene.append(gen1[i])
        else:
            baby_gene.append(gen2[i])
    return  np.array(baby_gene).astype(int)

def cloning(results,genes):
    indices = np.flip(np.argsort(results))
    cloned_genomes = []
    new_generation = []
    for i in range (0,6):
        #new_generation.append(genes[indices[i]])
        for j in range(0,2**(5-i)):
            cloned_genomes.append(genes[indices[i]])
    #creating_new_generation

    for i in range (0,20):
        number_1 = np.random.randint(low=0,high=63)
        number_2 = np.random.randint(low=0,high=63)
        new_generation.append(uniform_crossover(cloned_genomes[number_1],cloned_genomes[number_2]))
    #mutation of new_generation
    for i in range(0,5):
        ind_gen = np.random.randint(low=0,high=20)
        how_many = np.random.randint(low=1,high=4)
        for j in range (0,how_many):
            ind_gen_bit = np.random.randint(low=0,high=512)
            if (new_generation[ind_gen][ind_gen_bit] == 0):
                new_generation[ind_gen][ind_gen_bit] = 1
            else:
                 new_generation[ind_gen][ind_gen_bit] = 0
    return new_generation

def initiate_life(world,genes):
    for l in range(0,len(genes)):    
        world_1 = genes[l][calculate_N(world)]
        for i in range(1,100):
            world_1 = genes[l][calculate_N(world_1)]
            if (i>=90):
                genes_results[l] += fitness_func((world_1))
        genes_results[l] = genes_results[l]/10
    #plt.pcolor(world_1)
    #plt.colorbar()
    #plt.show()
    return genes_results

#%%SYMUALACJA 
world = random_create_world(50)
plt.pcolor(world)
plt.colorbar()
plt.show()
#initial cond
genes = []
results_generations = []

for i in range (0,20):
    genes.append(generate_random_genome())

for x in range (0,200):
    genes_results = np.zeros(len(genes))
    for k in range(0,10):      
        world = random_create_world(50) 
        genes_results +=initiate_life(world,genes)
    genes_results = genes_results/10
    new_result = np.sum(genes_results)
    results_generations.append(new_result)
    print(x,new_result)
    #print(np.sort(genes_results))
    genes = cloning(genes_results,genes)
    #print(fitness_func(world_1))
#%%RYSOWANIE KLATEK ANIMACJI DLA ZWYCIESKIEGO GENU
world = random_create_world(50) 

#plt.imshow(world,cmap='Greys')
#plt.savefig('intial'+'.png',dpi = 300, bbox_inches='tight', pad_inches = 0)
#plt.show()
for l in range(0,len(genes)):    
        world_1 = genes[l][calculate_N(world)]
        for i in range(1,100):
            world_1 = genes[l][calculate_N(world_1)] 
        plt.imshow(world_1,cmap='Greys')
        plt.title(genes[l])
        #plt.aspect('equal')
        #plt.savefig(str(l).rjust(3,'0')+'.png',dpi = 300, bbox_inches='tight', pad_inches = 0)
        #plt.colorbar()
        plt.show()
#%%Animacja
world = random_create_world(50) 
for i in range(1,200):

        plt.imshow(world,cmap='Greys')
        plt.title("time = " +str(i))
        #plt.aspect('equal')
        plt.savefig(str(i).rjust(3,'0')+'.png',dpi = 300, bbox_inches='tight')
        #plt.colorbar()
        plt.show()
        world = genes[-1][calculate_N(world)] 


#%%FITNESS PLOT
plt.plot(results_generations,'o')
plt.ylabel('Sum of fitness for all genes')
plt.xlabel('time')
#plt.savefig('Final_fitness.png',dpi = 300, bbox_inches='tight')
plt.show()
#%%
np.save('winners.npy', np.array(genes, dtype=object), allow_pickle=True)
winning = np.load('winners.npy', allow_pickle=True)
genes = np.array(winning).astype(int)
#%%

from PIL import Image

# List of PNG files
#png_files = ["001.png", "002.png", "003.png"]  # Add more file names as needed

# Open the first image to get the dimensions
first_image = Image.open("001.png")
# Create a GIF with the same dimensions as the first image
gif = Image.new(mode='RGB', size=first_image.size)
frames = []
# Iterate through each PNG file and append it to the GIF
for i in range (1,162):
    file_name = str(i).rjust(3,"0") +".png"
    png_image = Image.open(file_name)
    frames.append(png_image.convert("RGBA"))
alpha_channels = [frame.split()[-1] for frame in frames]

# Get the size of the first frame
width, height = frames[0].size

# Create a blank RGBA image
gif = Image.new("RGBA", (width, height))

# Paste each frame onto the blank image with its alpha channel
for frame, alpha_channel in zip(frames, alpha_channels):
    gif.paste(frame, (0, 0), mask=alpha_channel)

# Calculate duration per frame based on desired fps (e.g., 10 fps)
fps = 20
duration_per_frame = int(1000 / fps)  # Duration in milliseconds

# Save the frames as an animated GIF with adjusted fps
gif.save("output.gif", save_all=True, append_images=frames[1:], optimize=False, duration=duration_per_frame, loop=0)

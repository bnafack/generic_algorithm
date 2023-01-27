import pandas as pd 
import numpy as np
import pygad
import math
import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel('products.xlsx')
obj= np.array(df['Product'])
space = np.array(df['Space'])


function_inputs = space # Function space.
desired_output = 3 # Function output.
sol_per_population = int(math.pow(2,11))

def fitness_func(solution, solution_idx):
    # The fitness function calulates the sum of products between each input and its corresponding space.
    output = np.sum(solution*function_inputs)
    if (output>desired_output): 
        fitness=0.0
    else:
        # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
        fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness

def print_selected_obj(ga_instance,obj,space):
    solution,_,_ = ga_instance.best_solution()
    solution = np.array(solution).astype(float)
    print('\n')
    print(f'The optimal sum is {np.sum(solution*space)}')
    print('\n')
    print(f'object selected {obj[np.where(solution == 1)]}')
    print('\n')

def save_model(ga_instance):
    filename = 'genetic'
    ga_instance.save(filename=filename)

def plot_fitnes(ga_instance):
    ga_instance.plot_fitness(title="PyGAD with Adaptive Mutation", linewidth=5)


def show_best_genes(ga_instance):
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print('\n')
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    print('\n')

    #Let's save the model.  

    

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=100,
                       fitness_func=fitness_func,
                       num_parents_mating=2,
                       sol_per_pop=sol_per_population ,
                       num_genes=len(function_inputs),
                       mutation_type="adaptive",
                       mutation_num_genes=(3, 1),
                       gene_type=int,
                       init_range_low=0,
                       init_range_high=2,
                       parallel_processing=4,
                       stop_criteria=["saturate_7"])




if __name__=='__main__':
    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    print_selected_obj(ga_instance,obj,space)

    show_best_genes(ga_instance)
    save_model(ga_instance)
    plot_fitnes(ga_instance)
    
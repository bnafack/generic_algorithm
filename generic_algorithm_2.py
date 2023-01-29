import pandas as pd 
import numpy as np
import pygad
import math
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('products.xlsx')
obj= np.array(df['Product'])
space = np.array(df['Space'])
quantity= np.array(df['Quantity'])

gene_space=[range(i) for i in quantity ]

function_inputs = space # Function space.
desired_output = 4 # volume capacity we want to reach.
sol_per_population = int(math.pow(2,11)) # 2^11 number of sample in the initial population


def fitness_func(solution, solution_idx):
    """This function calculates the total fitness value by summing the product of 
    each input and its corresponding space.
    """
    output = np.sum(solution*function_inputs)
    if (output>desired_output): # discard a sample if the sum is more than the space available
        fitness=0.0
    else:
        # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
        fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness

def print_selected_obj(ga_instance,obj,space):

    """This function generates all van loading combinations 
    considering capacity, weight and size. It returns a list of possible combinations.
    """
    solution,_,_ = ga_instance.best_solution()
    solution = np.array(solution).astype(float)
    print('\n')
    print(f'The optimal sum is {np.sum(solution*space)}')
    print('\n')
    print(f'The maximun price is {np.sum(solution*quantity)}')
    print('\n')
    print(f'object selected {obj[np.where(solution != 0)]}')
    print('\n')



def save_model(ga_instance):
    filename = 'genetic'
    ga_instance.save(filename=filename)


def plot_fitnes(ga_instance):
    ga_instance.plot_fitness(title="PyGAD with Adaptive Mutation", linewidth=5)


def show_best_genes(ga_instance):
    """The function will identify the genes (inputs) with 
    the highest fitness value and display them as the best options.
    """
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print('\n')
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    print('\n')


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=100,
                       fitness_func=fitness_func,
                       num_parents_mating=2,
                       sol_per_pop=sol_per_population ,
                       num_genes=len(function_inputs),
                       mutation_type="adaptive",
                       mutation_num_genes=(3, 1),
                       gene_type=int,
                       parallel_processing=4,
                       gene_space = gene_space,
                       keep_elitism=2,
                       stop_criteria=["saturate_7"])




if __name__=='__main__':
    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    print_selected_obj(ga_instance,obj,space)

    show_best_genes(ga_instance)
    #Let's save the model.  

    save_model(ga_instance)
    plot_fitnes(ga_instance)
    
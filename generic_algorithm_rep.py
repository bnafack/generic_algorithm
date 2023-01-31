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
price = np.array(df['Price'])

gene_space=[range(i) for i in quantity ]

function_inputs = space # Function space.
desired_output = 5 # volume capacity we want to reach.
sol_per_population = int(math.pow(2,8)) # 2^11 number of sample in the initial population

    



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

def print_selected_obj(solution,obj,space):

    """This function generates print selected element to be load in the van.
    """
    solution = np.array(solution).astype(float)
    print('\n')
    print(f'The optimal sum is {np.sum(solution*space)}')
    print('\n')
    print(f'The maximun price is {np.sum(solution*price)}')
    print('\n')
    print(f'object selected {obj[np.where(solution != 0)]}')
    print('\n')



def save_model(ga_instance):
    filename = 'genetic'
    ga_instance.save(filename=filename)


def plot_fitnes(ga_instance):
    ga_instance.plot_fitness(title="PyGAD with Adaptive Mutation", linewidth=5)


def show_best_genes(solution):
    """The function show the best selected solution (inputs) with it
    fitness value and display them as the best options.
    """
    print('\n')
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fitness_func(solution,1)))

    print('\n')


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
def creat_instance():
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
    return ga_instance


def best_solution_with_max_price(number_of_iteration:int):
    """"This function repeatedly performs the GA algorithm for a specified number of iterations
    and returns the solution that yields the maximum price.
    The repetition is expected to converge due to the law of large numbers." """

    best_solution = []
    for _ in range(number_of_iteration):
        # Running the GA to optimize the parameters of the function.
        ga_instance=creat_instance()
        ga_instance.run()
        solution,_,_ = ga_instance.best_solution()
        solution = np.array(solution).astype(float)
        best_solution.append((solution,np.sum(solution*price)))
    
    #select the solution with the maximun price: 
    return sorted(best_solution,key=lambda x: x[1], reverse=True)[0][0]  




if __name__=='__main__':
    
    solution = best_solution_with_max_price(10)

    print_selected_obj(solution,obj,space)

    show_best_genes(solution)
    # plot_fitnes(ga_instance)
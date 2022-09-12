# Problem 4
## 4a
```c++
typedef struct{
    int immune = 0; // 0 if not immune, 1 if immune
    int infected = 0; // 0 if not infected, 1 if infected
    int days_infected = 0; // T
    int can_infect;
    int days_healthy;
    int num_interaction;
    int *interactions;
}person;

person *population = (person *)malloc(N*sizeof(person))
```
The struct allows me to easily access key details that are needed to see if the person is spreading or not. The `interactions` array contains the index of the people the person interacts with in the `population` array.


## 4b

```c++
int main(int argc, int *argv[]) {
    int N, T, X, f, graph_interaction, interactions_who;
    
    N = atoi(argv[1]); T = atoi(argv[2]) ; X = atoi(argv[3]); f = atoi(argv[4]);// Takes in the key numbers N, T, X and f, meaning number of points on graph, days after a person is infected that they start isolating, days of isolation needed before healthy and the probability of a sick person infecting another (from 0 to 100).
    extract_interactions_from_graph(&graph_interaction, &interactions_who, argv[5]); //find the amount of interactions each person has
    

    person *population = (person *)malloc(N*sizeof(person));
    for (int i; i<N; i++) {
        population[i].interactions = (int *)malloc(graph_interaction[i]*sizeof(int));
        population[i].num_interactions = graph_interaction[i];
        for (int j; j<graph_interaction[i]; j++)
            population[i].interactions[j] = interactions_who[i+j];
    }

    simulate(&population, N, T, X, f);

    return 0;
}

void simulate(**population, N, T, X, f) {
    person target;
    for (int i; i<N; i++) {
        target = (*population)[i];
        if (target.can_infect) {
            for (int j; i<target.num_interactions; j++) {
                if ((*population)[target.interactions[j]]).immune != 1 {
                    (*population)[target.interactions[j]]).infected = rand()/RAND_MAX*100 < f;
                }
            }
            target.days_infected++;
            target.can_infect = target.days_infected;
            if (target.days_infected > X+T) {
                target.can_infect = 0;
                target_immune = 1;
            }
        } else if (target.infected) {
            target.can_infect = 1;
        }
        (*population)[i] = target;
    }
}

```

## 4c
Since the `simulate` function is entirely dependent on the previous iteration, there is no point in parallelizing the actual function or simulation. Instead the parallelization would be smarter to apply to different versions of the simulations, meaning different combinations of $T$ and $f$. This means adding an additional for-loop outside of each simulation where each thread can be given a variety of combinations to try depending on the thread-to-combinations relathionship.
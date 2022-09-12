# Problem 1
## 1a

$T_{1, 1}$ must first be completed, only one worker can work. 

    Time spent: 1 hour

$T_{1, 2}$ and $T_{2, 1}$ must be completed next, both can be done simutaniously by two workers.

    Time spent: 2 hours

$T_{1, 3}$, $T_{2, 2}$ and $T_{3, 1}$ are next, all can be completed simutaniously.

    Time spent: 3 hours

$T_{1, 4}$, $T_{2, 3}$, $T_{3, 2}$ and $T_{4, 1}$ would be next, but only have three workers, $T_{5, 1}$ can wait. Does not matter which one waits.

    Time spent: 4 hours

$T_{1, 5}$, $T_{2, 4}$ and $T_{4, 1}$ are next. Make sure to include previously skipped ones.

    Time spent: 5 hours

$T_{3, 3}$, $T_{4, 2}$ and $T_{5, 1}$ can be completed.

    Time spent: 6 hours


The remaining tasks can be completed by following amount of workers:

1st iteration: 3 workers

    Time spent: 7 hours

2nd iteration: 3 workers

    Time spent: 8 hours

3rd iteration: 2 workers (due to dependencies)
    
    Time spent: 9 hours

4th iteration: 1 worker (due to dependencies)

    Time spent: 10 hours

5th iteration: 1 remaining task

    TIme spent 11 hours


## 1b

The maximum speedup can be achieved by having `5 workers`, because there at most can be five workers active. We find that this will take `9 hours` by following the shortest path of arrows and counting tasks. This means we will have a `2 hour` speedup.
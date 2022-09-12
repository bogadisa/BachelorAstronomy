# Problem 1
Worst case scenario the x- and y-array are too long to be kept in cache and must be loaded for every iteration of i. By the same logic also (using python slicing) the array `z[i:i+N]` is also too long and must be loaded for each iteration. That means the array x is, consisting of $8N$ bytes is load $N$ times; same goes for y. The z-array which is loaded for each iteration is of length $N$ and is loaded $N$ times, meaning it contributes equally to memory traffic as the x- and y-array does. Meaning the function for memory traffic would look like:

 > $f(x)=24N^2$

 
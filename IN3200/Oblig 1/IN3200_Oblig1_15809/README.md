# Oblig 1
### Candidate number: 15809
## Notes
- `read_graph_from_file.c` was by far the most challenging of the three files. Setteling for a less optimized version helped, but the implementation still has problems. After a lot of testing after trying to run `PageRank_iterations.c`, I concluded the way I allocate memory for the arrays causes it to crash later on if I try to allocate anything within the main-scope. After trying to fix this and understanding the problem, I figured it was easiest to implement a "wack" solution. This is why the `test` functions exist, as they let me allocate the scores list and so on.
- Each program has an easy to use print functions, that let you choose wether or not you want information about the calculations. Perfect for debugging and confirming results.
- For some reason `web-stanford.txt` does not work after `PageRank_iterations` has been parallelized. Even though that part of the program has nothing to do with the reading of the file
- While trying to parallelize `PageRank_iteration` I encountered what seems to be a system side overhead that inflates the time it takes my parallelized verion to complete calculations. Having nothing but a 10-iterations-long for loop which did easy work, the calculations took on average 12 milliseconds longer per for loop. This was after using `#pragma omp parallel for` inside the while loop, I did not have the same issue outside of the while-loop. Even though I understand there is overhead, this seemed like a very big difference. As I have compared timings with other IN3200 students, who have a nearly identical parallelization with seemingly no overhead on `100nodes_graph.txt`. I did not have access to another computer to run my code to see if its caused by my program or computer. Additional feedback is very welcome.

## Example of compilation and how to run the program
 - `mingw32-make all` to compile all programs.
 - Alternatively `mingw32-make filename.exe` to compile a single file. 
 - `.\"filename.exe"` to execute file of choosing.
 - `.\"main.exe"` requires additional arguments `"filename.txt" d epsilon top-n-websites` 

## Additional information
 - Depending on if you want to print results or not, remove // infront of the corresponding print functions within the header files. If you want to print every iteration you would need to change the if test
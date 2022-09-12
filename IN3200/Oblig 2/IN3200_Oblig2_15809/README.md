# Oblig 2
### Candidate number: 15809
## Notes
 - Serial version works perfectly
 - Parallel version I was unable to get to work. The program does not crash until it reaches the function for converting it back into a JPEG. This is most likely due to `MPI_Gatherv` not working as it should. Because of this I am unable to check if the program actually works otherwise. I belive there also is a chance my initial `MPI_Scatterv` also is the cause of the program crashing. Another problem might be the `u.image_data` not being allocated continuously.
 - I sometime use expressions such as `+ (my_rank!=0)` to not have to use a if test, as the example will return $1$ if its true and $0$ if false.
 - The program prints out where it is currently working, a great way to find where things go wrong.
 - The latest two slurm file can also be found in the `parallel_code` folder. One for each of the `MPI_Gatherv`.

 ## Example of compilation and how to run the program
 - `mingw32-make all` will compile everything in either of the two folders.
 - Alternatively `mingw32-make serial_main.exe` for just the main function.
 - Switching `serial` for `parallel` will work for compiling the parallel version.
 - `.\serial_main.exe 0.2 10 mona_lisa_noisy.jpg mona_lisa_denoised.jpg` to run the serial version with $\kappa=0.2$ and $n=10$ iterations.
 - The parallel version is ran by the command `sbatch job.txt` on the fox cluster. The default arguments are given in the example above and must be changed in the `job.txt` file if needed.
# Problem 5
The most obvious advantage of MPI is the controll over when you for example sync up information, which can be a cause of a lot of overhead with NUMA architecture. Furthermore MPI allows for selecting who you want to sync up with, making it easier for making groups of several threads work together on a single problem. The disadvantages of this however is the tediousnes of controlling who is talking to who and making sure there are no race-conditions or deadlocks. 

OpenMP implementations are easier to implement and understand. You dont have to worry about the nitty-gritty of communication. But OpenMP does not work on multisocket computers and so it wont be able to use all available CPUs .

The best implementation then, depending on number of CPUs per socket would be an OpenMp and MPI hybrid.n
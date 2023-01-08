### Sequential

Test MNIST training: 

| Run # | Time |
| --- | --- |
| 1 | 4718 |
| 2 | 4596 |
| 3 | 4562 |
| 4 | 4575 |
| 5 | 4514 |
| 6 | 4553 |
| 7 | 4467 |
| 8 | 4489 |
| 9 | 4483 |
| 10 | 4535 |

### MPI

Test MNIST training (always same due to MPI barrier): 

| Run # | number of processes $p$ | Time |
| --- | --- | --- |
| 1 | 1 | 4345 |
| 2 | 1 | 4215 |
| 3 | 1 | 4196 |
| 4 | 1 | 4244 |
| 5 | 1 | 4481 |
| 6 | 1 | 4242 |
| 7 | 1 | 4297 |
| 8 | 1 | 4217 |
| 9 | 1 | 4270 |
| 10 | 1 | 4297 |
| 1 | 2 | 2305 |
| 2 | 2 | 2569 |
| 3 | 2 | 2431 |
| 4 | 2 | 2414 |
| 5 | 2 | 2301 |
| 6 | 2 | 2624 |
| 7 | 2 | 2421 |
| 8 | 2 | 2361 |
| 9 | 2 | 2285 |
| 10 | 2 | 2377 |
| 1 | 4 | 2099 |
| 2 | 4 | 2237 |
| 3 | 4 | 1788 |
| 4 | 4 | 1947 |
| 5 | 4 | 1718 |
| 6 | 4 | 1695 |
| 7 | 4 | 1682 |
| 8 | 4 | 1663 |
| 9 | 4 | 1736 |
| 10 | 4 | 1843 |
| 1 | 8 | 1538 |
| 2 | 8 | 1653 |
| 3 | 8 | 1675 |
| 4 | 8 | 1758 |
| 5 | 8 | 1666 |
| 6 | 8 | 1616 |
| 7 | 8 | 1613 |
| 8 | 8 | 1646 |
| 9 | 8 | 1688 |
| 10 | 8 | 1663 |
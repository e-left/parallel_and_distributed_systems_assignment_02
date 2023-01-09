# make a 2d grid, a 3d grid, a 4d grid and a 5d grid
# coords: 0 up to 9

dims = [2, 3, 4, 5]
for d in dims:
    with open(f"datasets/dataset{d}.txt", "w+") as f:
        num = 0 
        while num < 10**d:
            points = [str(int((num / 10**x) % 10)) for x in range(d)]
            print(points)
            num += 1
            f.write(f"{', '.join(points)}\n")
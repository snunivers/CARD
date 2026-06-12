import random


def get_random_latin_square(n, min_iterations=None):
    
    id_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    
    cube = []
    for i in range(n):
        layer = [row[:] for row in id_matrix[i-n:]] + [row[:] for row in id_matrix[:i-n if i-n != 0 else n]]
        cube.append(layer)
    
    is_proper = True
    improper_cell = None
    min_iterations = min_iterations or n * n * n
    
    iteration = 0
    while iteration < min_iterations or not is_proper:
        t = [0, 0, 0]
        c = [0, 0, 0]
        
        if is_proper:
            
            while True:
                t = [
                    random.randint(0, n - 1),
                    random.randint(0, n - 1),
                    random.randint(0, n - 1)
                ]
                if cube[t[0]][t[1]][t[2]] == 0:
                    break
            
            
            for j in range(n):
                if cube[j][t[1]][t[2]] != 0:
                    c[0] = j
                    break
            for j in range(n):
                if cube[t[0]][j][t[2]] != 0:
                    c[1] = j
                    break
            for j in range(n):
                if cube[t[0]][t[1]][j] != 0:
                    c[2] = j
                    break
        else:
            t = improper_cell[:]
            candidates = [[], [], []]
            
            for j in range(n):
                if cube[j][t[1]][t[2]] == 1:
                    candidates[0].append(j)
                if cube[t[0]][j][t[2]] == 1:
                    candidates[1].append(j)
                if cube[t[0]][t[1]][j] == 1:
                    candidates[2].append(j)
            
            c[0] = random.choice(candidates[0])
            c[1] = random.choice(candidates[1])
            c[2] = random.choice(candidates[2])
        
        
        cube[t[0]][t[1]][t[2]] += 1
        cube[t[0]][c[1]][c[2]] += 1
        cube[c[0]][c[1]][t[2]] += 1
        cube[c[0]][t[1]][c[2]] += 1
        cube[t[0]][t[1]][c[2]] -= 1
        cube[t[0]][c[1]][t[2]] -= 1
        cube[c[0]][t[1]][t[2]] -= 1
        cube[c[0]][c[1]][c[2]] -= 1
        
        is_proper = cube[c[0]][c[1]][c[2]] != -1
        if not is_proper:
            improper_cell = c[:]
        
        iteration += 1
    
    
    square = [[0] * n for _ in range(n)]
    for x in range(n):
        for y in range(n):
            for s in range(n):
                if cube[x][y][s] == 1:
                    square[x][y] = s
                    break
    
    return square


if __name__ == "__main__":
    n = 8
    square = get_random_latin_square(n)
    
    print(f"Randomly generated {n}x{n} Latin square:")
    for row in square:
        print(" ".join(f"{x:2d}" for x in row))

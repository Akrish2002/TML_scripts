from concurrent.futures import ProcessPoolExecutor

def square(n):
    return n * n

if __name__ == '__main__':
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(square, range(10)))
    print(results)
    
    #Create a default process pool
    pool = multiprocessing.Pool()  

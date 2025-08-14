import functions as fn
from time import sleep
        
def main():
        w, b = 0, 0
        for _ in range(1000):
                w, b = fn.full_batch(w, b, learning_rate=0.01)
                
        fn.plot(w, b)
      #  fn.stochastic()
      #  fn.mini_batch()

if __name__ == "__main__":
        main()


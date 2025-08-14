import functions as fn
        
def main():
    w, b, MSE_loss_old = 0, 0, 0
    MSE_losses = []
    streak = 1
    
    while True:
        w, b, MSE_loss_new = fn.full_batch(w, b, learning_rate=0.01)        # Replace fn.full_batch with fn.stochastic() or fn.mini_batch() -- mini batch has batch_size parameter too! batch size is between 1 and len(data) = 7, inclusive

        if MSE_loss_new == MSE_loss_old:
            streak += 1

        else:
            streak = 1

        MSE_loss_old = MSE_loss_new
        MSE_losses.append(MSE_loss_old)

        
        # Stop after last 5 MSE Loss values are the same OR # iterations >= 100,000 
        # Latter needed for stochastic SGD where last 5 MSE Loss are usually never the same
        if streak == 5 or len(MSE_losses) >= 100000:
            break
        
    # Plot the data points and the final regression line
    fn.plot(w, b)

    # Plot the MSE loss curve
    fn.plot_MSE(MSE_losses)

if __name__ == "__main__":
    main()



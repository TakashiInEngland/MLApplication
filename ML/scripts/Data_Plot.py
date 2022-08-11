
def data_plot(DF,target_var,set_input_var,algorithm,screen):

    import matplotlib.pyplot as plt
    from matplotlib import style
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from pathlib import Path

    currentFilePath = os.path.realpath(__file__)
    parentPath = str(Path(currentFilePath).parents[0])
    scriptPath = os.path.join(parentPath,'scripts')
    os.chdir(scriptPath)
    from MB import SP_model_building
    
    y_train,y_train_predicted,y_actual,y_predicted = SP_model_building(DF,target_var,set_input_var,algorithm)
    
    style.use("ggplot")
    
    y_actual1 = np.array(y_actual)
    y_predicted1 = np.array(y_predicted)
    
    y_train1 = np.array(y_train)
    y_train_predicted1 = np.array(y_train_predicted)
    
    # For test data
    axmin = y_actual1.min()
    axmax = y_actual1.max()
    
    # Update the minimal value
    for z in [y_actual1,y_predicted1]:
        if axmin> z.min():
            axmin = z.min()
        if axmax < z.max():
            axmax = z.max()
    
    # For training data
    axmin_train = y_train1.min()
    axmax_train = y_train1.max()
    
    # Update the minimal value
    for z in [y_train1,y_train_predicted1]:
        if axmin_train> z.min():
            axmin_train = z.min()
        if axmax_train < z.max():
            axmax_train = z.max()
    
    # Draw a scatter plot
    fig = plt.figure(figsize=(12,6))
    plt.suptitle(algorithm, x=0.5, y=1.05, fontsize=20,verticalalignment='top')
    gs = gridspec.GridSpec(2,2) 
    
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,0])
    ax4 = plt.subplot(gs[1,1])
    
    # Draw a plot
    ax1.plot(y_train,color="#00A3E0",label="Actual")
    ax1.plot(y_train_predicted,color="#183A54",label="Predicted")
    #ax1.set_title("Train Data Plot, " + output_variable, fontsize=15)
    ax1.set_ylabel("Train Data",fontsize = 12)
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(0,1.02,1,0.102),loc=3,
               ncol=2,borderaxespad=0)
    
    # Draw a parity plot
    ax2.plot(y_train1, y_train_predicted1, 'o', color = 'navy')
    #ax2.set_title('Train Parity Plot, ' + output_variable, fontsize = 15) 
    ax2.set_xlabel('Train Actual', fontsize = 12)
    ax2.set_ylabel('Predicted', fontsize = 12)
    ax2.grid(True)
    # Define the edge scale of a scatter plot
    #ax1.xlim([axmin, axmax])
    #ax1.ylim([axmin, axmax])
    # Draw a diagnal line
    ax2.plot([axmin_train, axmax_train], [axmin_train, axmax_train], color = "gray")
    
    # Draw a plot
    ax3.plot(y_actual,color="#00A3E0",label="Actual")
    ax3.plot(y_predicted,color="#183A54",label="Predicted")
    #ax3.set_title("Test Data Plot, " + output_variable, fontsize=15)
    ax3.set_ylabel("Test Data",fontsize = 12)
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(0,1.02,1,0.102),loc=3,
               ncol=2,borderaxespad=0)
    # Draw a parity plot
    ax4.plot(y_actual1, y_predicted1, 'o', color = 'navy')
    #ax4.set_title('Train Parity Plot, ' + output_variable, fontsize = 15) 
    ax4.set_xlabel('Test Actual', fontsize = 12)
    ax4.set_ylabel('Predicted', fontsize = 12)
    ax4.grid(True)
    # Define the edge scale of a scatter plot
    #ax1.xlim([axmin, axmax])
    #ax1.ylim([axmin, axmax])
    # Draw a diagnal line
    ax4.plot([axmin, axmax], [axmin, axmax], color = "gray")
    
    #fig.tight_layout()
    
    #plt.legend()
    canvas2 = FigureCanvasTkAgg(fig,screen)
    canvas2.show()
    canvas2.get_tk_widget().pack()  
    #plt.plot([axmin, axmax], [axmin + threshold, axmax + threshold], color = "lightgray", label = 'threshold = 5ppm')
    #plt.plot([axmin, axmax], [axmin - threshold, axmax - threshold] , color = "lightgray")
    
    #plt.fill_between([axmin, axmax],[axmin- threshold, axmax- threshold] , [axmin+ threshold, axmax+ threshold] , color='gray',alpha=0.25)
    
    #plt.legend(loc='upper left')


# GUI Development
import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg


currentFilePath = os.path.realpath(__file__)
parentPath = str(Path(currentFilePath).parents[0])
scriptPath = os.path.join(parentPath,'scripts')
dataPath = os.path.join(parentPath,'data')
os.chdir(scriptPath)
from FS_updated import feature_selection
from model_building_updated import model_building
from Data_Plot import data_plot
#from MB import SP_model_building

#from Model_Building import model_building
#from Feature_Selection import feature_selection
#from Feature_Importance_Visualisation import FIV

# Font
EXTRA_L_FONT = ("Verdana",14)
LARGE_FONT = ("Verdana",12)
NORM_FONT = ("Verdana",10)
SMALL_FONT = ("Verdana",8)
style.use("ggplot")

def tutorial():   
    def page2():
        tut.destroy()
        tut2 = tk.Tk()
        
        def page3():
            tut2.destroy()
            tut3 = tk.Tk()
            
            tut3.wm_title("Part 3")
            
            label = tk.Label(tut3,text="Part 3",font=NORM_FONT)
            label.pack(side="top",fill="x",pady=10)
            B1 = ttk.Button(tut3,text="Done!",command=tut3.destroy)
            B1.pack()
            tut3.mainloop()
        
        tut2.wm_title("Part 2")
        label = tk.Label(tut2,text="Part 2",font=NORM_FONT)
        label.pack(side="top",fill="x",pady=10)
        B1 = ttk.Button(tut2,text="Next!",command=page3)
        B1.pack()
        tut2.mainloop()
    
    def ChatBot():
        tut.destroy()
        chatbot = tk.Tk()
        chatbot.geometry("240x120")
        
        chatbot.wm_title("ChatBot")
        label = tk.Label(chatbot,text="Talk with ChatBot",fg="#183A54",font=NORM_FONT)
        label.pack(side="top",fill="x",pady=10)
        chatbot.mainloop()
    
    tut = tk.Tk()
    tut.geometry("240x120")
    tut.wm_title("Tutorial")
    label = tk.Label(tut,text="What do you need help with?",font=NORM_FONT)
    label.pack(side="top",fill="x",pady=10)
    
    B1 = ttk.Button(tut,text="Overview of the application",command=page2)
    B1.pack()
    
    B2 = ttk.Button(tut,text="What is Machine Learning?",command=lambda:popupmsg("Not yet completed"))
    B2.pack()
    
    B3 = ttk.Button(tut,text="Talk with Chatbot",command=ChatBot)
    B3.pack()
    
    tut.mainloop()

def popupmsg(msg):
    popup = tk.Tk()
    
    #def leavemini():
    #    popup.destroy()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

# def animate(i):
    
#    datalist = pulldata.split("\n")
#    xList = []
#    yList = []
#    for eachLine in datalist:
#        if len(eachLine) > 1:
#            x,y = eachLine.split(",")
#            xList.append(int(x))
#            yList.append(int(y))
            
#    a.clear()  # Get rid of everything on a subplot
#    a.plot(xList,yList)
    
class MLapp(tk.Tk):
    
    def __init__(self,*args,**kwargs):   # it is a method to initialise, when Class is called, it initialises contents
        
        tk.Tk.__init__(self,*args,**kwargs)
        
        #iconfile = parentPath + "Image/images.png"
        #tk.Tk.iconbitmap(self,default=iconfile)
        tk.Tk.wm_title(self, "Machine Learning Application")
        
        container = tk.Frame(self)
        container.pack(side="top",fill="both",expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=1)    # tearoff makes a menubar deattachable and move it around on the screen
        filemenu.add_command(label="Save Settings",command=lambda:popupmsg("Not Supported Just Yet"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit",command=self.quit())
        menubar.add_cascade(label="File",menu=filemenu)
        
        # Help 
        Help = tk.Menu(menubar, tearoff=1)
        Help.add_command(label="Tutorial",
                         command=tutorial)
        menubar.add_cascade(label="Help",menu=Help)
        
        tk.Tk.config(self,menu=menubar)
        
        self.frames = {}
        
        for F in (StartPage,LogIn,ChooseData):#PageOne, PageTwo, PageThree):
        
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(StartPage)
        
    def show_frame(self,cont):
        
        frame = self.frames[cont]
        frame.tkraise()
        
    """def storage_data(self,storage):
        
        if storage == True:
            To_store_df = storage
            
        elif storage == False:
            return To_read_df
     """
    
class HoverButton(tk.Button):
    
    def __init__(self, master, **kw):
        tk.Button.__init__(self,master=master,**kw)
        self.defaultBackground = "#1a8ad8"#self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self['activebackground']

    def on_leave(self, e):
        self['background'] = self.defaultBackground
       
class StartPage(tk.Frame):
    
    def __init__(self,parent,controller):
        
        tk.Frame.__init__(self,parent)
        #tk.Tk.configure(self,background="#183A54")
        #tk.Tk.title(self,"MACHINE LEARNING APPLICATION")
        label = tk.Label(self, text="Use this application at your own risk.\nThere is neither promise nor warranty.",fg="#183A54",font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        #1a8ad8
        #icon1 = PhotoImage("C:/Users/1523724/Desktop/Application/yes.png")

        button1 = tk.Button(self,text="Agree",bg="#1a8ad8",fg="#edeeef",activebackground="#157ac1",width=9,#fg="#183A54",bg="#a1abcd",#image=icon1,compound=TOP,
                            command=lambda:controller.show_frame(LogIn))
                            #command=qf)
                            #command=lambda: qf2("see this works"))
        #button1.image = icon1
        button1.place(x=160,y=110)
        
        #icon2 = PhotoImage("C:/Users/1523724/Desktop/Application/no.png")
        button2 = tk.Button(self,text="Disagree",bg="#d7dfe5",fg="#427ab2",width=9,#fg="#183A54",bg="#a1abcd",#image=icon2,compound=TOP,
                            command=Error)
                            #lambda:controller.show_frame(PageTwo))
        #button2.image = icon2
        button2.place(x=260,y=110)
def LogInError():
    messagebox.showinfo("Message","Have Entered an invalid username or password.")
    
def LogInSuccess():
    messagebox.showinfo("Message","Logined successfully.")
    
def ValError():
    messagebox.showinfo("Message","Have entered invalid information.")

class LogIn(tk.Frame):
    
    def __init__(self,parent,controller):
        
        tk.Frame.__init__(self,parent)
        loginLabel = tk.Label(self, text="Log in",fg="#183A54",font=EXTRA_L_FONT)
        loginLabel.place(x=120,y=20)
        
        usernameLabel = tk.Label(self, text="Username: ",fg="#183A54",font=NORM_FONT)
        usernameLabel.place(x=120,y=70)
        var = StringVar()
        usernameB = ttk.Entry(self,textvariable=var)
        usernameB.place(x=200,y=70)
        
        passwordLabel = tk.Label(self, text="Password: ",fg="#183A54",font=NORM_FONT)
        passwordLabel.place(x=120,y=110)
        var2 = StringVar()
        bullet = "\u2022"
        passwordB = ttk.Entry(self,textvariable=var2,show=bullet)#show="*")
        passwordB.place(x=200,y=110)
        
        # Scrollbar
        #scrollbar = ttk.Scrollbar(self,orient=VERTICAL)
        #scrollbar.place()
        # Read data
        acountDF = pd.read_excel(dataPath + "password.xlsx")
        aL = acountDF["username"].tolist()
        def clear_text():
            usernameB.delete(0,"end")
            passwordB.delete(0,"end")
        
        def confirm():
            
            eUsername = usernameB.get()
            ePassword = passwordB.get()
            if eUsername in aL:
                index_for = acountDF.index[acountDF["username"] == eUsername][0]
                password = acountDF.iloc[index_for,1]
                if ePassword == password:
                    LogInSuccess()
                    controller.show_frame(ChooseData)
                
            #for item in acountList["username"]:
            #    if eUsername in aL and acountList[eUsername] == ePassword:
            #        print("Correct")
                    
                else:
                    LogInError()
            else:
                LogInError()
        
        def ResPass():
            
            tut = tk.Tk()
            tut.wm_title("Reset Password")
            tut.geometry("240x200")
            
            UserLabel = tk.Label(tut,text="Username",fg="#183A54",font=NORM_FONT)
            UserLabel.place(x=11,y=50)
            var3 = StringVar()
            username_ = ttk.Entry(tut,textvariable=var3)
            username_.place(x=80,y=50)
            
            VarLabel = tk.Label(tut,text="Answer",fg="#183A54",font=NORM_FONT)
            VarLabel.place(x=20,y=100)
            var4 = StringVar()
            VarGuess = ttk.Entry(tut,textvariable=var4)
            VarGuess.place(x=80,y=100)
            
            def Pass_Val():
                
                VarUsername = username_.get()
                VarGuess_ = VarGuess.get()
                
                def Show_Pass():
                    
                    tut.destroy()
                    tut2 = tk.Tk()
                    tut2.wm_title("Show Password")
                    tut2.geometry("250x200")
                    
                    password_to = acountDF.iloc[index_for,1]
                    text1 = tk.Label(tut2,text="Your password is {}".format(password_to),font=LARGE_FONT,fg="#183A54")
                    text1.place(x=11,y=50)
                    
                    ChangePass = tk.Label(tut2,text="I strongly recommend you change",font=SMALL_FONT)
                    ChangePass.place(x=10,y=100)
                    
                    ChangePass2 = tk.Label(tut2,text="your password immediately",font=SMALL_FONT)
                    ChangePass2.place(x=10,y=118)
                    
                    val5 = StringVar()
                    To_what = ttk.Entry(tut2,textvariable=val5)
                    To_what.place(x=10,y=150)
                    
                    def Change_Pass():
                        
                        newP = To_what.get()
                        acountDF.iloc[index_for,1] = newP
                        acountDF.to_excel(dataPath + "password.xlsx",index=None)
                        messagebox.showinfo("Message","Password successfully changed.")
                        
                    CHB = tk.Button(tut2,text="Change",bg="#1a8ad8",fg="#edeeef",width=9,command=Change_Pass)
                    CHB.place(x=140,y=148)
                    
                if VarUsername in aL:
                    index_for = acountDF.index[acountDF["username"] == VarUsername][0]
                    varif = acountDF.iloc[index_for,2]
                    if VarGuess_ == varif:
                        Show_Pass()
                        
                    else:
                        ValError()
                else:
                    ValError()
                #tut2.Text(text=)
                
            ShowB = tk.Button(tut,text="Done",bg="#1a8ad8",fg="#edeeef",width=9,command=Pass_Val)
            ShowB.place(x=80,y=150)
        
        #LoginButton = ttk.Button(self,text="Login",style="BW.TLabel",command=lambda:confirm())
        LoginButton = tk.Button(self,text="Login",bg="#1a8ad8",fg="#edeeef",width=9,command=lambda:confirm())
        LoginButton.place(x=150,y=160)
        
        #CancelButton = ttk.Button(self,text="Clear",style="BW.TLabel",command=clear_text)
        CancelButton = tk.Button(self,text="Clear",bg="#d7dfe5",fg="#427ab2",width=9,command=clear_text)
        CancelButton.place(x=230,y=160)
        
        RestButton = ttk.Button(self,text="Forgot password?",style="green/black.TLabel",command=ResPass)
        RestButton.place(x=330,y=111)
        
#green/black.TLabel
#BW.TLabel
def Empty_Value():
    messagebox.showinfo("Message","Please choose a value.")  
    mainloop()
        
class ChooseData(tk.Frame):
    
    def __init__(self,parent,controller):
        
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Data Selection",fg="#183A54",font=EXTRA_L_FONT)
        label.pack(pady=10,padx=10)
        
        #def HeatMap():
        
        def callback_data():
            
            import os
            import tkinter
            root = tk.Tk()
            root.withdraw()
            
            file_Type = [("","*")]
            
            iniDir = os.path.abspath(os.path.dirname(dataPath))
            messagebox.showinfo("Data Selection","Please choose a data file.")
            fileName = filedialog.askopenfilename(filetypes=file_Type,initialdir=iniDir)
            
            def FileError():
                messagebox.showinfo("Message","Have chosen an invalid format. \n Only CSV/ XLSX Aceepted.")
                mainloop()
            
            if fileName.endswith(".csv"):
                To_read_DF = pd.read_csv(fileName)
                #print(To_read_DF)
            elif fileName.endswith(".xlsx"):
                To_read_DF = pd.read_excel(fileName)
                #print(To_read_DF)
            else:
                FileError()
            
            #controller.storage_data(To_read_DF)
            #Data = To_read_DF
            fN = fileName.split("/")

            ret = messagebox.askyesno("Confirmation","You have chosen " + fN[-1] + "\nAre you sure?")
            
            
            """def Explore_Data(data):
                
                tut0 = tk.Tk()
                tut0.wm_title("Data Exploartion")
                tut0.geometry("240x200")
                
                titleL0 = tk.Label(tut, text="Let's Explore Data!",fg="#183A54",font=NORM_FONT)
                titleL0.place(x=50,y=20)
           """ 
            def Vis_PrePage(data):
                
                tut = tk.Tk()
                tut.wm_title("Data Visualisation")
                tut.geometry("240x200")
                
                titleL = tk.Label(tut, text="Let's Visualise Data!",fg="#183A54",font=LARGE_FONT)
                titleL.place(x=50,y=20)
                
                def Heat_Map():
                    
                    import seaborn as sns 
                    
                    tut2 = tk.Tk()
                    tut2.wm_title("Heatmap")
                    tut2.geometry("700x650")
                    
                    titleL2 = tk.Label(tut2, text="Heatmap",fg="#183A54",font=EXTRA_L_FONT)
                    titleL2.pack()
                    
                    closeB = tk.Button(tut2,text="Close",bg="#d7dfe5",fg="#427ab2",width=9,command=lambda:tut2.destroy())
                    closeB.pack()
                    #closeB.place(x=400,y=360)
                                     
                    corr = To_read_DF.corr()
                    corr = corr.round(2)
                    
                    cm = sns.light_palette("green", as_cmap=True)
                    fig, ax = plt.subplots(figsize=(10,8))   
                    sns.heatmap(corr,cmap=cm,annot=True,ax=ax)
                    fig.tight_layout()
                    
                    canvas = FigureCanvasTkAgg(fig,tut2)
                    canvas.show()
                    canvas.get_tk_widget().pack()
                
                button0 = ttk.Button(tut,text="Heatmap",command=Heat_Map)
                button0.place(x=40,y=60)
                
                def Two_Parameter():
                    
                    tut01 = tk.Tk()
                    tut01.wm_title("Variable Comparison")
                    tut01.geometry("400x300")
                    
                    titleL01 = tk.Label(tut01, text="Two Variable Comparison",fg="#183A54",font=EXTRA_L_FONT)
                    titleL01.pack()
                    
                    closeB_01 = tk.Button(tut01,text="Close",bg="#d7dfe5",fg="#427ab2",width=9,command=lambda:tut01.destroy())
                    closeB_01.pack()
                    #closeB.place(x=400,y=360)
                                     
                
                def Raw_Data():
                    
                    tut02 = tk.Tk()
                    tut02.wm_title("Variable Comparison")
                    tut02.geometry("400x300")
                    
                    titleL02 = tk.Label(tut01, text="Plot Raw Data",fg="#183A54",font=EXTRA_L_FONT)
                    titleL02.pack()
                    
                    closeB_02 = tk.Button(tut01,text="Close",bg="#d7dfe5",fg="#427ab2",width=9,command=lambda:tut02.destroy())
                    closeB_02.pack()
                
                def Hist_Gram():
                    tut3 = tk.Tk()
                    tut3.wm_title("Histogram")
                    tut3.geometry("500x400")
                    
                    titleL3 = tk.Label(tut3, text="Histogram",fg="#183A54",font=EXTRA_L_FONT)
                    titleL3.pack()
                    
                    variable = StringVar(tut3)
                    
                    DF_col = To_read_DF.columns.tolist()
                    #variable.set("--Select--") # Default value 
                    
                    
                    
                    def remove_plot(canvas):
                        canvas.get_tk_widget().destroy()
                    
                    def Reflect_CH(what):
                        
                        try:
                        
                            #if Bin_Chosen == "":
                            #    nu = 10 # Default it 10 binnings 
                            #else:
                            #    nu = Bin_Chosen
                            chosen_ = To_read_DF[what]
                            fig2 = plt.figure()
                            plt.hist(chosen_,bins=10)
                            
                            canvas2 = FigureCanvasTkAgg(fig2,tut3)
                            canvas2.show()
                            canvas2.get_tk_widget().pack()
                            
                            tut3.bind_all('<Button-1>',lambda event:remove_plot(canvas2))
                            tut3.mainloop()
                        except:
                            pass
                    
                    closeB_ = ttk.Button(tut3,text="Close",command=lambda:tut3.destroy())
                    closeB_.pack() 
                    
                    w = ttk.OptionMenu(tut3, variable,"--Select--",*DF_col,command=Reflect_CH)
                    w.pack()
                    
                    BinCH_B = tk.Label(tut3,text="Change # of Bins",fg="#183A54",font=SMALL_FONT)
                    BinCH_B.pack()
                        
                    var = StringVar()
                    BinCH_N = Spinbox(tut3,values=(10,20,30,40,50),command=Reflect_CH(w))
                    BinCH_N.pack()
                
                button1 = ttk.Button(tut,text="Histogram",command=Hist_Gram)
                button1.place(x=120,y=60)
                
                button_001 = ttk.Button(tut,text="X-Y",command=Two_Parameter)
                button_001.place(x=40,y=90)
                
                button_002 = ttk.Button(tut,text="Raw",command=Raw_Data)
                button_002.place(x=120,y=90)
                
                #def erase_page():
                #    tut.destroy()
                
                button2 = tk.Button(tut,text="<Back",bg="#d7dfe5",fg="#427ab2",width=9,
                                     command=lambda:tut.destroy())
                button2.place(x=80,y=150)
                
                
                ## To choose a target variable from available columns
                def Variable_Selection():
                    # to choose X variables and y variable for feature selection and model building
                    
                    tut4 = tk.Tk()                
                    tut4.wm_title("Target Variable Selection")
                    tut4.geometry("370x300")
                    
                    titleL4 = tk.Label(tut4, text="Choose a Target Variable",fg="#183A54",font=LARGE_FONT)
                    titleL4.pack()
                    
                    #canvas = tk.Canvas(tut4)
                    
                    #scrollBar0 = ttk.Scrollbar(tut4,orient="vertical")
                    #scrollBar0.pack(side=RIGHT,fill=Y)
                    btn_4 = StringVar(tut4)
                    
                    rb_txt = To_read_DF.columns.tolist()
                    for i in range(len(rb_txt)):
                        ttk.Radiobutton(tut4,text=rb_txt[i],value=rb_txt[i],variable=btn_4).place(x=50,y=70+(i*24))
                    
                    #scrollBar.config(command=canvas.yview)
                    #scrollBar.pack(side="right",fill="y")
                    #canvas.pack(fill='both', expand=True)
                             
                    ## To choose a feature selection algorithm
                    def Feature_Selection():
                    
                        #if btn_4 == None:
                            
                        #    Empty_Value()

                        """try:
                            tut.destroy()
                            tut2.destroy()
                            tut3.destroy()
                        except:
                            pass
                        """
                        #else:
                            
                        tut5 = tk.Tk()                
                        tut5.wm_title("Feature Selection")
                        tut5.geometry("500x400")
                        
                        titleL5 = tk.Label(tut5, text="Feature Selection",fg="#183A54",font=EXTRA_L_FONT)
                        titleL5.pack()
                        
                        Fs_method = ["Ridge","Lasso","XGBoost","Linear Regressor","Random Forest Regressor","Gradient Boosting Regressor","Extra Tree Regressor"]
                        
                        btn = StringVar(tut5)
                        
                        for i in range(len(Fs_method)):
                            ttk.Radiobutton(tut5,text=Fs_method[i],value=Fs_method[i],variable=btn).place(x=50,y=70+(i*24))
                    
                        ### To show a feature selection result
                        def Show_FS_Result():
            
                            # The chosen feature selection algorithm
                            Chosen_FS_al = btn.get()
                            # The target variable 
                            target_variable_name = btn_4.get()
                            
                            tut6 = tk.Tk()                
                            tut6.wm_title("Feature Selection Result")
                            tut6.geometry("500x400")
                            
                            titleL6 = tk.Label(tut6, text="Show Coefficients of Parameters",fg="#183A54",font=EXTRA_L_FONT)
                            titleL6.pack()
                            
                            # Use the pipeline code to get the feature selection result
                            sr = feature_selection(To_read_DF,Chosen_FS_al,target_variable_name)
                            
                            
                            listFrame = Frame(tut6)
                            listFrame.pack()
                            
                            list_var1 = tk.StringVar()
                            
                            scrollBar = Scrollbar(listFrame)
                            scrollBar.pack(side=RIGHT,fill=Y)
                            
                            tut6.listBox = tk.Listbox(listFrame,listvariable=list_var1,selectmode=EXTENDED,width=50)
                            
                            #tut4.listBox.place(x=10,y=80)                       
                            
                            feature_im = sr.index.tolist()

                            for iin,c in enumerate(feature_im):
                                coeff = sr.iloc[iin,0]
                                coeff = '%.3f'%(coeff)
                                integrated = c + "/  " +str(coeff)
                                tut6.listBox.insert(iin,integrated) 
                            tut6.listBox.pack()
                            scrollBar.config(command=tut6.listBox.yview)
                            tut6.listBox.config(yscrollcommand=scrollBar.set)
                            
                            #tframe = Frame(tut6).pack()
                            
                            #dict_data = To_read_DF.to_dict()
                            #print(dict_data)
                            
                            #listFrame = Frame(tut4)
                            #listFrame.pack()
                            
                            #scrollBar = Scrollbar(listFrame)
                            #scrollBar.pack(side=RIGHT,fill=Y)
                            
                            #tut4.listBox = tk.Listbox(listFrame,listvariable=list_var1,selectmode=EXTENDED)
                    
                            #tut4.listBox.place(x=10,y=80)
                            def Model_Building():
                                
                                #try:
                                #    tut6.destroy()
                                #except:
                                #    pass
                                
                                #titleL6 = tk.Label(tut6, text="Building models...Wait a minute!",fg="#183A54",font=NORM_FONT)
                                #titleL6.pack(side="bottom")
                                
                                cV = tut6.listBox.curselection()
                                cV_ = [feature_im[i] for i in cV]
                                #print(cV_)
                                        
                                tut7 = tk.Tk()                
                                tut7.wm_title("Model Building Result")
                                tut7.geometry("300x400")
                                # geometry(width x height)
                                
                                titleL7 = tk.Label(tut7, text="Compare RMSE of Models",fg="#183A54",font=EXTRA_L_FONT)
                                titleL7.pack()
                                
                                RMSE_df = model_building(To_read_DF,target_variable_name,cV_)
                                
                                btn_7 = StringVar(tut7)
                                alg_sr = RMSE_df.iloc[:,0]
                                
                                tk.Label(tut7,text="Train",fg="#183A54",font=SMALL_FONT).place(x=100,y=45)
                                tk.Label(tut7,text="Test",fg="#183A54",font=SMALL_FONT).place(x=170,y=45)    
                                         
                                def Parity_Plot():
                                    
                                    Al_for_plot = btn_7.get()
                                    
                                    tut9 = tk.Tk()  
                                    title_9 = "Plots - " + Al_for_plot
                                    tut9.wm_title(title_9)
                                    tut9.geometry("700x430")
                                    
                                    #titleL9 = tk.Label(tut9, text="Show Plots",fg="#183A54",font=EXTRA_L_FONT)
                                    #titleL9.pack()
                                    
                                    
                                    
                                    data_plot(To_read_DF,target_variable_name,cV_,Al_for_plot,tut9)
                                    
                                
                                btn_7 = StringVar(tut7)
                                for i in range(len(alg_sr)):
                                    
                                    ttk.Radiobutton(tut7,text=alg_sr[i],value=alg_sr[i],variable=btn_7).place(x=50,y=70+(i*30))
                                    
                                    Bo_t = ttk.Entry(tut7,width=7)
                                    rmse_v_t = RMSE_df.iloc[i,1]
                                    rmse_v_t = '%.2f'%(rmse_v_t)
                                    Bo_t.insert(tkinter.END,rmse_v_t)
                                    Bo_t.place(x=100,y=70+(i*30))         
                                    
                                    Bo = ttk.Entry(tut7,width=7)
                                    rmse_v = RMSE_df.iloc[i,2]
                                    rmse_v = '%.2f'%(rmse_v)
                                    Bo.insert(tkinter.END,rmse_v)
                                    Bo.place(x=170,y=70+(i*30))
                                    
                               
                                
                                ttk.Button(tut7,text="Plot",command=lambda:Parity_Plot()).place(x=150,y=70+((len(alg_sr)+1)*30))
                            
                                
                               
                            #table.show()
                            button20 = ttk.Button(tut6,text="Done",
                                                command=lambda:Model_Building())
                            button20.pack()
                
                        
                        
                        
                        button10 = tk.Button(tut5,text="Okay",bg="#d7dfe5",fg="#427ab2",width=9,
                                            command=lambda:Show_FS_Result())
                        button10.pack()
                        
                    #scrollBar0.config(command=tut4.yview)
                    
                    move_to_fs = tk.Button(tut4,text="Okay",bg="#d7dfe5",fg="#427ab2",width=9,
                                           command=lambda:Feature_Selection())          
                    move_to_fs.pack()
                
                button3 = tk.Button(tut,text="Next>",bg="#1a8ad8",fg="#edeeef",width=9,
                                    command=lambda:Variable_Selection())
                button3.place(x=155,y=150)
            
            if ret == True:
                #def nextgo():
                Vis_PrePage(To_read_DF)
            else:
                root.mainloop()
                controller.show_frame(ChooseData)
                
            return fileName
        button1 = ttk.Button(self,text="Open File",command=callback_data)
        
        button1.place(x=220,y=100)

"""
# No need to have a screen to choose algorithms for model building, as it does try all algorithms and 
# Show all resutls so that users can choose which algorithm to go with
class AlgoPage(tk.Frame):
    
    def __init__(self,parent,controller):
            
            tk.Frame.__init__(self,parent)
            label = tk.Label(self, text="Choose Algorithms for Model Building",fg="#183A54",font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            chk_txt = ["MLP Regressor","Linear Regressor","XGBoost","Ridge","SVR","Lasso","Gradient Boosting Regressor","Random Forest Regressor"]
            chk_bln = {}
            
            for i in range(len(chk_txt)):
                chk_bln[i] = tk.BooleanVar()
                chk = tk.Checkbutton(self,variable=chk_bln[i],text=chk_txt[i])
                chk.place(x=50,y=30+(i*24))        
            
            button0 = ttk.Button(self,text="Okay>",command=lambda:controller.show_frame(ParityPage))
            button0.place(x=575,y=300)
            
            button1 = ttk.Button(self,text="<Back",command=lambda:controller.show_frame(FSPage))
            button1.place(x=500,y=300)
"""    
    
def Error():
    messagebox.showinfo("Message","I'am afraid to tell you\n you can't use this application without agreement.")        
    
app = MLapp()
app.geometry("500x250")#("1280x720")
#ani = animation.FuncAnimation(f,animate,interval=5000)
app.mainloop()

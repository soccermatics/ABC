#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:53:20 2021

@author: davsu428
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

Nr=[]
br_index=[]

ABC=np.zeros((3,3))
AUCC=np.zeros((3,3))

from pylab import rcParams
rcParams['figure.figsize'] = 14/2.54, 14/2.54
matplotlib.font_manager.FontProperties(family='Helvetica',size=11)

fig,ax=plt.subplots(3,3)
fig2,ax2=plt.subplots(3,3)
fig3,ax3=plt.subplots(3,3)
fig4,ax4=plt.subplots(3,3)
fig5,ax5=plt.subplots(3,3)


cj=-1

#List of the scores for each of the 9 datasets
score_positives=[]
score_negatives=[]

#Parameters in the model
mean_differences = [5,3,1]
amount_of_spurious_data= [100,1000,10000]


#Boolean to indicate if we include B_-, B_+ and F1.
include_other_curves=False


########################################################################
#Create the dataset and calculate AUC curve
#
#Positive class have mean value 10 and standard deviation 2.
#The 'hard' members of the negative class have mean value 10-meandiff and standard deviation 2.
#The 'easy' members of the negative class have mean value 2 and standard deviation 2.
#sizeofspurious is the number of 'easy' members.
########################################################################
for meandiff in mean_differences:
  cj=cj+1
  ci=-1
  for sizeofspurious in amount_of_spurious_data:#range(1,5001,100):
    ci=ci+1
    
    valn=np.append(np.random.normal(10-meandiff,2,1000),np.random.normal(2,2,sizeofspurious))
    valp=np.random.normal(10,2,1000)
    
    #Create training set and perform logistic regression
    Y_train=np.append(-1*np.ones(len(valn)),1*np.ones(len(valp))).reshape(-1, 1)
    X_train=np.append(valn,valp).reshape(-1, 1)
    model = skl_lm.LogisticRegression(solver='lbfgs')
    model.fit(X_train, Y_train) 
    
    #Calculate the model predictions for the positive and negative classes
    #Calculate log-likelihoods for those scores.
    predict_prob = model.predict_proba(X_train)
    pp=predict_prob[:,0]
    p = -np.log(pp[Y_train[:,0]==1]/(1-pp[Y_train[:,0]==1]))
    n = -np.log(pp[Y_train[:,0]==-1]/(1-pp[Y_train[:,0]==-1]))
    allvals = -np.log(pp/(1-pp))

    #Now make a histogram of scores (figure 1)
    ncounts, bins = np.histogram(n,bins=np.arange(-15,10,1))
    pcounts, bins = np.histogram(p,bins=np.arange(-15,10,1))
    if cj==0:
        ax2[ci, cj].set_ylabel('Frequency')
    if (cj==1 and ci==2):
        ax2[ci, cj].set_xlabel("Score") 
    if ci==2:
        ax2[ci, cj].set_xticks(np.arange(-10,10,step=5))
    else:
        ax2[ci, cj].set_xticks(np.arange(-20,10,step=100))
    
    
    ax2[ci, cj].bar(bins[1:],ncounts,color='g', alpha=0.3)
    ax2[ci, cj].bar(bins[1:],pcounts,color='y', alpha=0.7)
    ax2[ci, cj].set_xlim((-11,9))
    if ci==2:
        ax2[ci, cj].set_ylim((0,1750))
        if cj==0:
            ax2[ci, cj].set_yticks(np.arange(0,1501,step=500))
        else:
            ax2[ci, cj].set_yticks(np.arange(-100,1501,step=5000))
        ax2[ci, cj].text(5,1350,'(%s)' % chr(97+3*ci+cj)) 
    else:
        ax2[ci, cj].set_ylim((0,550))
        if cj==0:
            ax2[ci, cj].set_yticks(np.arange(0,501,step=250))
        else:
            ax2[ci, cj].set_yticks(np.arange(-100,1501,step=5000))
        ax2[ci, cj].text(5,400,'(%s)' % chr(97+3*ci+cj)) 
    
    
    score_positives=score_positives + [p]
    score_negatives=score_negatives + [n]

#%%

for cj in [0,1,2]:
  for ci in [0,1,2]:
    ########################################################################
    #Plot ROC and Precision-Recall curve
    #
    #Caluate Sensitivity, Fbr_index and Tbr_index from data.
    # Used to make ROC and sensitivity-specificity curves.
    # Rvals is the range of the threshold.
    ########################################################################

    p=score_positives[cj*3+ci]
    n=score_negatives[cj*3+ci]
    
    rvals=np.arange(-10,10,0.1)
    Sensitivity=np.zeros(len(rvals))
    FPR=np.zeros(len(rvals))
    TPR=np.zeros(len(rvals))
    
    #Loop over the r thresholds measuring each value
    for j,r in enumerate(rvals):
        FP = (n>r) 
        TP = (p>r)    
        Sensitivity[j]=sum(np.array(TP))/(sum(np.array(FP))+sum(np.array(TP)))
        FPR[j]=sum(np.array(FP))/len(n)
        TPR[j]=sum(np.array(TP))/len(p)
    
    
    ########################################################################
    #Calculate AUC using sampling method and plot ROC
    ########################################################################
    
    AUC=np.zeros(len(rvals))
    sample_size=1000
    #Pick negative
    neg_score=np.random.choice(n,sample_size)
    #Pick positive
    pos_score=np.random.choice(p,sample_size)
    AUC = sum(pos_score>neg_score)/sample_size
    
    #Calculate AUC using integral method
    AUC_analytic=sum(-np.diff(FPR)*TPR[1:])
    
    
    #Plot the ROC curve (figure 2)
    ax3[ci, cj].plot(FPR, TPR,color='k')
    ax3[ci, cj].set_xlim((0.0,1.0))
    ax3[ci, cj].set_ylim((0,1.1))
    ax3[ci, cj].text(0.6,0.1,'A=%.3f' % AUC)
    ax3[ci, cj].spines['top'].set_visible(False)
    ax3[ci, cj].spines['right'].set_visible(False)
    ax3[ci, cj].text(0.8,0.8,'(%s)' % chr(97+3*ci+cj)) 
    if ci==2:
        ax3[ci, cj].set_xticks(np.arange(0,1.1,step=0.5))
    else:
        ax3[ci, cj].set_xticks(np.arange(-20,10,step=100))
    if cj==0:
        ax3[ci, cj].set_yticks(np.arange(0,1.1,step=0.5))
    else:
        ax3[ci, cj].set_yticks(np.arange(-20,10,step=100))
    if ((ci==1) & (cj==0)):
        ax3[ci, cj].set_ylabel('True Positive Rate: $v(r)$')
    if ((ci==2) & (cj==1)):
        ax3[ci, cj].set_xlabel("False Positive Rate: $u(r)$")   
    
    
    #Calculate area under precision-recall curve.
    #This figure is not in the paper.
    fullPrecRecall=np.nansum(-np.diff(TPR)*Sensitivity[1:])
    
    ax4[ci, cj].plot(TPR,Sensitivity,color='k')
    ax4[ci, cj].set_xlim((0.0,1.0))
    ax4[ci, cj].set_ylim((0,1.1))
    ax4[ci, cj].text(0.6,0.1,'%.3f' % fullPrecRecall )
    ax4[ci, cj].spines['top'].set_visible(False)
    ax4[ci, cj].spines['right'].set_visible(False)
    
    if ((ci==1) & (cj==0)):
        ax4[ci, cj].set_ylabel('Recall')
    if ((ci==2) & (cj==1)):
        ax4[ci, cj].set_xlabel("True Positive Rate: $v(r)$")         
    ax4[ci, cj].text(0.8,0.8,'(%s)' % chr(97+3*ci+cj)) 
    
    

    ########################################################################
    #Calculate B-test using sampling method and integral 
    ########################################################################

    p=score_positives[cj*3+cj]
    n=score_negatives[cj*3+cj]    
    #This is the B(r) in the article.
    B_test=np.zeros(len(rvals))
    #This is the B_+(r) in the article.
    B_test_plus=np.zeros(len(rvals))
    #This is the B_-(r) in the article.
    B_test_minus=np.zeros(len(rvals))
    
    
    
    rAUC=np.zeros(len(rvals))
    N=len(n)
    P=len(p)
    NN=N+P
    
    #Create an B curve using sampling method.
    for j,r in enumerate(rvals):
        
    
        #Pick true negative samples
        neg_score=np.random.choice(n,sample_size)
        #Pick true positive samples
        pos_score=np.random.choice(p,sample_size)
        
        
        #Find all the positively labelled values
        poslabel = np.array(list(n[n>r]) + list(p[p>r]))
        #Check to see if there are members in this set and then caluclate B(r)
        #for each threshold. 
        if poslabel.any():
            pos_label_score=np.random.choice(poslabel,sample_size)
            B_test[j]= sum(pos_score>=pos_label_score)/sample_size
        else:
            #Sometimes there are no values, then just set to NaN.
            B_test[j]= np.nan
    
        #Find all the negatively labelled values
        neglabel = np.array(list(n[n<=r]) + list(p[p<=r]))
     
        Nlen=len(n[n>r])
        Plen=len(p[p>r])
        
        #The reason for the if statements here
        #is that sometimes there aren't enough observations 
        #to do the test. 
        if Nlen>sample_size:
            #Pick positive greater than r
            pos_score_r=np.random.choice(p[p>r],sample_size)
            #Pick negative greater than r
            neg_score_r=np.random.choice(n[n>r],sample_size)
            B_test_plus[j] = Plen*np.sum(pos_score>pos_score_r)/((Plen+Nlen)*sample_size)
            pos_score=np.random.choice(p,sample_size)
            B_test_minus[j] = Nlen*np.sum(pos_score>neg_score_r)/((Plen+Nlen)*sample_size)
            
        elif Plen>sample_size:
            pos_score_r=np.random.choice(p[p>r],sample_size)
            B_test_plus[j] = Plen*np.sum(pos_score>pos_score_r)/((Plen+Nlen)*sample_size) 
            B_test_minus[j] = np.nan
        else:
            B_test_plus[j] = np.nan
            B_test_minus[j] = np.nan
        
        #An AUC for all observations with score greater than r. 
        rAUC[j]=sum(-np.diff(FPR[j:])*TPR[j+1:])
    
    #This is figures 3 and 5 in the paper
    #Here we calculate it analyticly using equation from paper.
    B_test_analytic = ((P*np.power(TPR,2)/2)+ N*rAUC) /(TPR*P + N*FPR)
    
    if include_other_curves:
        #Plot B_+(r)
        ax5[ci, cj].plot(rvals, (P*np.power(TPR,2)/2)/(TPR*P + N*FPR),color='y')    
        #Plot B_-(r)
        ax5[ci, cj].plot(rvals, (N*rAUC) /(TPR*P + N*FPR),color='g')
        #Plot F1(r)
        ax5[ci, cj].plot(rvals, 2*Sensitivity*TPR/(Sensitivity+TPR),color='r')    #ax.plot(B_test_analytic, rAUC,color='r')
    #Plot B(r)
    ax5[ci, cj].plot(rvals, B_test_analytic,color='k')
    ax5[ci, cj].plot([-15,10], [0.5, 0.5],color='k',linestyle=':')
    ax5[ci, cj].set_xlim((-10,7.5))
    ax5[ci, cj].set_ylim((0,1))
    ax5[ci, cj].spines['top'].set_visible(False)
    ax5[ci, cj].spines['right'].set_visible(False)
    ax5[ci, cj].text(-10,0.9,'(%s)' % chr(97+3*ci+cj)) 
    
    if ci==2:
        ax5[ci, cj].set_xticks(np.arange(-10,10,step=5))
    else:
        ax5[ci, cj].set_xticks(np.arange(-20,10,step=100))
    if cj==0:
        ax5[ci, cj].set_yticks(np.arange(0,1.1,step=0.5))
    else:
        ax5[ci, cj].set_yticks(np.arange(-20,10,step=100))
        
    if ((ci==1) & (cj==0)):
        ax5[ci, cj].set_ylabel('$B(r)$')
    if ((ci==2) & (cj==1)):
        ax5[ci, cj].set_xlabel("Threshold: $r$")     
    
    
    rmin=np.nansum(B_test>0.5)
    br_index = br_index + [rmin]    


#%%
########################################################################
#Plot B-test against FPR, TPR and Sensitivity
########################################################################
    
    #This is figure 4 in the paper.
    ax[ci, cj].plot(B_test_analytic, FPR,color='r')
    ax[ci, cj].plot(B_test_analytic, TPR,color='b')
    ax[ci, cj].plot(B_test_analytic, Sensitivity,color='k')    
    ax[ci, cj].plot([0.4, 0.4], [0, 1],color='k',linestyle=':')
    ax[ci, cj].plot([0.6, 0.6], [0, 1],color='k',linestyle=':')
    ax[ci, cj].plot([0.5, 0.5], [0, 1],color='k')
    ax[ci, cj].set_xlim((0.0,1.0))
    ax[ci, cj].set_ylim((0,1.1))
    ax[ci, cj].spines['top'].set_visible(False)
    ax[ci, cj].spines['right'].set_visible(False)
    if ((ci==1) & (cj==0)):
        ax[ci, cj].set_ylabel('Recall: $C(r)$')
    if ((ci==2) & (cj==1)):
        ax[ci, cj].set_xlabel(" $B(r)$") 

    
    ax[ci, cj].text(0.1,0.8,'(%s)' % chr(97+3*ci+cj)) 

    if ci==2:
        ax[ci, cj].set_xticks(np.arange(0,1.1,step=0.5))
    else:
        ax[ci, cj].set_xticks(np.arange(-20,10,step=100))
    if cj==0:
        ax[ci, cj].set_yticks(np.arange(0,1.1,step=0.2))
    else:
        ax[ci, cj].set_yticks(np.arange(-20,10,step=100))

    
    max_F1=np.nanargmax(2*Sensitivity*TPR/(Sensitivity+TPR))
    rbalance=np.nansum(B_test_analytic>0.5)

    print(max_F1)
    print(rbalance)

    lower=0.4
    upper=0.6
    
    forty=np.nansum(B_test_analytic>upper)
    sixty=np.nansum(B_test_analytic>lower)
    
    #Store the integral of area between 40 and 60
    #Just an idea.
    ABC[ci,cj]= (np.nansum(-np.diff(B_test_analytic[forty:sixty])*TPR[forty+1:sixty]) - np.nansum(-np.diff(B_test_analytic[forty:sixty])*FPR[forty+1:sixty]))/(upper-lower)
    
    #Gives sensitivity at r_b
    #ax[ci, cj].text(0.62,0.8,'%.3f' % Sensitivity[rmin] )
    
    #Store AUC
    AUCC[ci,cj]=AUC_analytic
    
    
    rmin=np.nansum(B_test_analytic>0.5)
    ax2[ci, cj].plot([rvals[rmin], rvals[rmin]], [0, 1000],color='k',linestyle=':')
    ax2[ci, cj].spines['top'].set_visible(False)
    ax2[ci, cj].spines['right'].set_visible(False)
    
    outofbalance=(rAUC-(P*TPR*(1-TPR)+N*FPR)/(2*N))

    Nr = Nr + [rmin]
    

plt.show()
    
fig.savefig('Balance.pdf')

fig3.savefig('AUC.pdf')

if include_other_curves:
    fig5.savefig('Bplot2.pdf')
else:
    fig5.savefig('Bplot.pdf')
    
fig2.savefig('Distribution.pdf')

fig4.savefig('PrecisionRecall.pdf')






    